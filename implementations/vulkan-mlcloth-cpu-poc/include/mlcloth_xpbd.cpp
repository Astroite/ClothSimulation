#include "mlcloth_xpbd.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <utility>

namespace mlcloth {
namespace {

constexpr float kSafeDistance = 1.0e-9f;
constexpr float kSafeTwiceArea = 1.0e-12f;
constexpr float kDenominatorFloor = 1.0e-20f;

inline float dot3(const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

inline void cross3(const float* a, const float* b, float* out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

inline float length3(const float* a) { return std::sqrt(dot3(a, a)); }

inline float alpha_tilde(float compliance, float timestep) {
    const float squared = timestep * timestep;
    return compliance / std::max(squared, 1.0e-12f);
}

}  // namespace

float capsule_signed_distance(const float point[3], const Capsule& capsule) {
    const float offset[3] = {
        point[0] - capsule.centre[0],
        point[1] - capsule.centre[1],
        point[2] - capsule.centre[2],
    };
    const float projection = std::clamp(dot3(offset, capsule.axis), -capsule.halfLength, capsule.halfLength);
    const float radial[3] = {
        offset[0] - capsule.axis[0] * projection,
        offset[1] - capsule.axis[1] * projection,
        offset[2] - capsule.axis[2] * projection,
    };
    return length3(radial) - capsule.radius;
}

std::vector<uint32_t> derive_bend_pairs(uint32_t vertexCount, const uint32_t* triangles, uint32_t triangleCount) {
    // Which triangles own each undirected edge. A manifold edge has two; a boundary edge
    // has one and contributes no bend constraint, because there is no second corner.
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> owners;
    for (uint32_t triangle = 0; triangle < triangleCount; ++triangle) {
        const uint32_t corner[3] = {
            triangles[3 * triangle], triangles[3 * triangle + 1], triangles[3 * triangle + 2],
        };
        for (int k = 0; k < 3; ++k) {
            const uint32_t a = corner[k], b = corner[(k + 1) % 3];
            owners[{ std::min(a, b), std::max(a, b) }].push_back(triangle);
        }
    }
    std::set<std::pair<uint32_t, uint32_t>> unique;
    for (const auto& entry : owners) {
        if (entry.second.size() != 2) continue;
        uint32_t opposite[2] = { vertexCount, vertexCount };
        for (int side = 0; side < 2; ++side) {
            const uint32_t triangle = entry.second[size_t(side)];
            for (int k = 0; k < 3; ++k) {
                const uint32_t corner = triangles[3 * triangle + k];
                if (corner != entry.first.first && corner != entry.first.second) opposite[side] = corner;
            }
        }
        if (opposite[0] >= vertexCount || opposite[1] >= vertexCount) continue;
        if (opposite[0] == opposite[1]) continue;
        unique.insert({ std::min(opposite[0], opposite[1]), std::max(opposite[0], opposite[1]) });
    }
    // Sorted and deduplicated, so the ordering is a property of the mesh rather than of
    // iteration order: the calibrator indexes its target lengths by exactly this list.
    std::vector<uint32_t> pairs;
    pairs.reserve(unique.size() * 2);
    for (const auto& pair : unique) {
        pairs.push_back(pair.first);
        pairs.push_back(pair.second);
    }
    return pairs;
}

bool XpbdSolver::build(uint32_t vertexCount,
                       const uint32_t* edgePairs, uint32_t edgeCount, uint32_t bendStart,
                       const uint32_t* triangles, uint32_t triangleCount,
                       const float* vertexMassKg,
                       const uint32_t* pinMask,
                       const float* targetLengthM,
                       const float* targetAreaM2,
                       std::string& err) {
    err.clear();
    if (vertexCount == 0 || edgeCount == 0) {
        err = "the solver needs at least one vertex and one edge";
        return false;
    }
    if (!edgePairs || !vertexMassKg || !pinMask || !targetLengthM) {
        err = "a required solver input is null";
        return false;
    }
    if (triangleCount != 0 && (!triangles || !targetAreaM2)) {
        err = "triangles were given without target areas, or the reverse";
        return false;
    }

    if (bendStart > edgeCount) {
        err = "bendStart is past the end of the constraint list";
        return false;
    }
    vertexCount_ = vertexCount;
    constraintCount_ = edgeCount;
    bendStart_ = bendStart;
    triangleCount_ = triangleCount;

    pairs_.assign(edgePairs, edgePairs + size_t(edgeCount) * 2);
    for (uint32_t index = 0; index < edgeCount * 2; ++index) {
        if (pairs_[index] >= vertexCount) {
            err = "an edge indexes a vertex outside the mesh";
            return false;
        }
    }
    triangles_.assign(triangles ? triangles : edgePairs, (triangles ? triangles : edgePairs) + size_t(triangleCount) * 3);
    for (uint32_t index = 0; index < triangleCount * 3; ++index) {
        if (triangles_[index] >= vertexCount) {
            err = "a triangle indexes a vertex outside the mesh";
            return false;
        }
    }

    targetLength_.assign(targetLengthM, targetLengthM + edgeCount);
    for (uint32_t edge = 0; edge < edgeCount; ++edge) {
        if (!std::isfinite(targetLength_[edge]) || targetLength_[edge] < 0.0f) {
            err = "a target length is negative or not finite";
            return false;
        }
    }
    targetArea_.assign(targetAreaM2, targetAreaM2 + triangleCount);
    for (uint32_t triangle = 0; triangle < triangleCount; ++triangle) {
        if (!std::isfinite(targetArea_[triangle]) || targetArea_[triangle] < 0.0f) {
            err = "a target area is negative or not finite";
            return false;
        }
    }

    // A pinned vertex gets zero inverse mass. That is what makes it immovable inside the
    // passes; the closing overwrite with its guide value is a separate mechanism, and both
    // are needed -- zero inverse mass alone would freeze it at the previous frame.
    inverseMass_.assign(vertexCount, 0.0f);
    pinned_.assign(vertexCount, 0);
    for (uint32_t vertex = 0; vertex < vertexCount; ++vertex) {
        const uint32_t flag = pinMask[vertex];
        if (flag > 1) {
            err = "the pin mask is not boolean";
            return false;
        }
        pinned_[vertex] = static_cast<uint8_t>(flag);
        const float mass = vertexMassKg[vertex];
        if (!(mass > 0.0f) || !std::isfinite(mass)) {
            err = "a vertex mass is not positive and finite";
            return false;
        }
        inverseMass_[vertex] = flag ? 0.0f : 1.0f / mass;
    }

    averaging_.assign(vertexCount, 0.0f);
    minEdge_.assign(vertexCount, std::numeric_limits<float>::infinity());
    for (uint32_t edge = 0; edge < edgeCount; ++edge) {
        const uint32_t a = pairs_[2 * edge], b = pairs_[2 * edge + 1];
        averaging_[a] += 1.0f;
        averaging_[b] += 1.0f;
        // The trust scale is the vertex's *own* shortest constraint, not the mesh's. A
        // global minimum lets the single shortest edge anywhere in the garment dictate the
        // clamp everywhere, which on an irregular tessellation is most of it.
        minEdge_[a] = std::min(minEdge_[a], targetLength_[edge]);
        minEdge_[b] = std::min(minEdge_[b], targetLength_[edge]);
    }
    for (uint32_t vertex = 0; vertex < vertexCount; ++vertex) {
        if (!std::isfinite(minEdge_[vertex])) minEdge_[vertex] = 0.0f;
    }

    areaIncident_.assign(vertexCount, 0.0f);
    for (uint32_t triangle = 0; triangle < triangleCount; ++triangle) {
        for (int corner = 0; corner < 3; ++corner) {
            areaIncident_[triangles_[3 * triangle + corner]] += 1.0f;
        }
    }

    multiplier_.assign(edgeCount, 0.0f);
    areaMultiplier_.assign(triangleCount, 0.0f);
    guideMultiplier_.assign(vertexCount, 0.0f);
    accumulator_.assign(size_t(vertexCount) * 3, 0.0f);
    scratch_.assign(size_t(vertexCount) * 3, 0.0f);

    configure(config_);
    return true;
}

void XpbdSolver::configure(const XpbdConfig& config) {
    config_ = config;
    const float stretchAlpha = alpha_tilde(config_.stretchCompliance, config_.timestep);
    const float bendAlpha = alpha_tilde(config_.bendCompliance, config_.timestep);
    weightSum_.assign(constraintCount_, 0.0f);
    alpha_.assign(constraintCount_, 0.0f);
    denominator_.assign(constraintCount_, 0.0f);
    alive_.assign(constraintCount_, 0);
    for (uint32_t edge = 0; edge < constraintCount_; ++edge) {
        const uint32_t a = pairs_[2 * edge], b = pairs_[2 * edge + 1];
        const float sum = inverseMass_[a] + inverseMass_[b];
        const float alpha = edge < bendStart_ ? stretchAlpha : bendAlpha;
        weightSum_[edge] = sum;
        alpha_[edge] = alpha;
        denominator_[edge] = std::max(sum + alpha, kDenominatorFloor);
        // Both endpoints pinned means the constraint can never do anything, and letting it
        // accumulate a multiplier would be a slow drift with no corresponding motion.
        alive_[edge] = sum > 0.0f ? 1 : 0;
    }
}

void XpbdSolver::apply_stretch(const float* current, float* next) {
    std::fill(accumulator_.begin(), accumulator_.end(), 0.0f);
    for (uint32_t edge = 0; edge < constraintCount_; ++edge) {
        const uint32_t a = pairs_[2 * edge], b = pairs_[2 * edge + 1];
        // Formed in the stored endpoint order, never "mine minus theirs".
        const float delta[3] = {
            current[3 * a] - current[3 * b],
            current[3 * a + 1] - current[3 * b + 1],
            current[3 * a + 2] - current[3 * b + 2],
        };
        const float distance = length3(delta);
        if (!(distance > kSafeDistance) || !alive_[edge]) continue;
        const float inverse = 1.0f / distance;
        const float gradient[3] = { delta[0] * inverse, delta[1] * inverse, delta[2] * inverse };

        float residual = distance - targetLength_[edge];
        if (config_.oneSided) residual = std::max(residual, 0.0f);
        if (residual > 0.0f) ++stats_.stretchActive;

        const float numerator = -residual - alpha_[edge] * multiplier_[edge];
        const float step = numerator / denominator_[edge];
        multiplier_[edge] += step;

        // The endpoint-dependent sign is applied to the correction, after the multiplier
        // update, never inside it.
        for (int axis = 0; axis < 3; ++axis) {
            const float contribution = gradient[axis] * step;
            accumulator_[3 * a + axis] += contribution;
            accumulator_[3 * b + axis] -= contribution;
        }
    }
    for (uint32_t vertex = 0; vertex < vertexCount_; ++vertex) {
        const float scale = pinned_[vertex]
            ? 0.0f
            : config_.relaxation * inverseMass_[vertex] / std::max(averaging_[vertex], 1.0f);
        for (int axis = 0; axis < 3; ++axis) {
            next[3 * vertex + axis] = current[3 * vertex + axis] + scale * accumulator_[3 * vertex + axis];
        }
    }
}

void XpbdSolver::apply_area(const float* current, float* next) {
    std::fill(accumulator_.begin(), accumulator_.end(), 0.0f);
    for (uint32_t triangle = 0; triangle < triangleCount_; ++triangle) {
        const uint32_t index[3] = {
            triangles_[3 * triangle], triangles_[3 * triangle + 1], triangles_[3 * triangle + 2],
        };
        const float* a = &current[3 * index[0]];
        const float* b = &current[3 * index[1]];
        const float* c = &current[3 * index[2]];
        const float ab[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
        const float ac[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
        float normal[3];
        cross3(ab, ac, normal);
        const float twiceArea = length3(normal);
        if (!(twiceArea > kSafeTwiceArea)) continue;
        const float inverse = 1.0f / twiceArea;
        const float unit[3] = { normal[0] * inverse, normal[1] * inverse, normal[2] * inverse };

        const float target = config_.areaFloor * targetArea_[triangle];
        const float residual = std::max(target - 0.5f * twiceArea, 0.0f);
        if (residual <= 0.0f) continue;
        ++stats_.areaFloorsActive;

        // grad_a A = 0.5 * (b - c) x n, and cyclically. These sum to zero, as a constraint
        // that cannot move the centre of mass must.
        const float bc[3] = { b[0] - c[0], b[1] - c[1], b[2] - c[2] };
        const float ca[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
        const float abv[3] = { a[0] - b[0], a[1] - b[1], a[2] - b[2] };
        float gradient[3][3];
        cross3(bc, unit, gradient[0]);
        cross3(ca, unit, gradient[1]);
        cross3(abv, unit, gradient[2]);
        for (int corner = 0; corner < 3; ++corner) {
            for (int axis = 0; axis < 3; ++axis) gradient[corner][axis] *= 0.5f;
        }

        // Gradients are not unit vectors here, so the denominator is sum_i w_i |grad_i|^2.
        // Summed in corner order 0, 1, 2 -- fixed by the mesh -- so all three vertices of a
        // triangle produce the same multiplier update.
        float denominator = 0.0f;
        for (int corner = 0; corner < 3; ++corner) {
            denominator += inverseMass_[index[corner]] * dot3(gradient[corner], gradient[corner]);
        }
        if (!(denominator > 0.0f)) continue;

        const float alpha = alpha_tilde(config_.areaCompliance, config_.timestep);
        const float numerator = -residual - alpha * areaMultiplier_[triangle];
        const float step = numerator / std::max(denominator + alpha, kDenominatorFloor);
        areaMultiplier_[triangle] += step;

        // grad C = -grad A while the floor is violated, so the correction inflates the
        // triangle. Each vertex applies only the gradient of the corner it occupies.
        for (int corner = 0; corner < 3; ++corner) {
            for (int axis = 0; axis < 3; ++axis) {
                accumulator_[3 * index[corner] + axis] += -gradient[corner][axis] * step;
            }
        }
    }
    for (uint32_t vertex = 0; vertex < vertexCount_; ++vertex) {
        const float scale = pinned_[vertex]
            ? 0.0f
            : config_.relaxation * inverseMass_[vertex] / std::max(areaIncident_[vertex], 1.0f);
        for (int axis = 0; axis < 3; ++axis) {
            next[3 * vertex + axis] = current[3 * vertex + axis] + scale * accumulator_[3 * vertex + axis];
        }
    }
}

void XpbdSolver::apply_guide(const float* guide, float* position) {
    const float alpha = alpha_tilde(config_.guideCompliance, config_.timestep);
    for (uint32_t vertex = 0; vertex < vertexCount_; ++vertex) {
        if (pinned_[vertex]) continue;
        const float weight = inverseMass_[vertex];
        if (!(weight > 0.0f)) continue;

        // Zero confidence means "no guide at all", so it maps to an infinite alpha-tilde
        // rather than a large one, and the step is masked rather than divided -- dividing
        // would give inf and then nan through 0 * inf.
        const float confidence = confidence_.empty() ? 1.0f : confidence_[vertex];
        if (!(confidence > 0.0f)) continue;
        const float alphaTilde = alpha / std::max(confidence, 1.0e-6f);

        const float delta[3] = {
            position[3 * vertex] - guide[3 * vertex],
            position[3 * vertex + 1] - guide[3 * vertex + 1],
            position[3 * vertex + 2] - guide[3 * vertex + 2],
        };
        const float distance = length3(delta);
        if (!(distance > kSafeDistance)) continue;

        const float inverse = 1.0f / distance;
        // C(x) = |x - g|, so grad C is a unit vector and grad C M^-1 grad C^T is just w.
        const float numerator = -distance - alphaTilde * guideMultiplier_[vertex];
        const float step = numerator / std::max(weight + alphaTilde, kDenominatorFloor);
        guideMultiplier_[vertex] += step;
        // The normal points from the guide towards the vertex and the step is negative, so
        // this moves the vertex towards the guide. Confidence acts through alpha-tilde
        // only; it is deliberately NOT a factor on the correction, which would be a
        // different function -- it would scale the step without changing the compliant
        // equilibrium the accumulated multiplier converges to.
        for (int axis = 0; axis < 3; ++axis) {
            position[3 * vertex + axis] += weight * delta[axis] * inverse * step;
        }
    }
}

void XpbdSolver::set_collision_mask(const uint8_t* mask) {
    if (mask == nullptr) { collisionMask_.clear(); return; }
    collisionMask_.assign(mask, mask + vertexCount_);
}

void XpbdSolver::apply_contacts(const std::vector<Capsule>& capsules, float* position) {
    if (capsules.empty()) return;
    for (uint32_t vertex = 0; vertex < vertexCount_; ++vertex) {
        if (!(inverseMass_[vertex] > 0.0f)) continue;
        if (!collisionMask_.empty() && !collisionMask_[vertex]) continue;
        float* point = &position[3 * vertex];
        // The deepest capsule wins. Resolving every capsule in turn would let one push the
        // vertex into another and report both as resolved.
        const Capsule* deepest = nullptr;
        float worst = 0.0f;
        for (const Capsule& capsule : capsules) {
            const float signed_distance = capsule_signed_distance(point, capsule) - config_.contactOffset;
            if (signed_distance < worst) {
                worst = signed_distance;
                deepest = &capsule;
            }
        }
        if (!deepest) continue;
        ++stats_.contactsResolved;

        // Push out along the outward radial direction. Unlike a stored half-plane normal
        // this is recomputed from the current position, so it is never a stale plane.
        const float offset[3] = {
            point[0] - deepest->centre[0],
            point[1] - deepest->centre[1],
            point[2] - deepest->centre[2],
        };
        const float projection = std::clamp(dot3(offset, deepest->axis), -deepest->halfLength, deepest->halfLength);
        float radial[3] = {
            offset[0] - deepest->axis[0] * projection,
            offset[1] - deepest->axis[1] * projection,
            offset[2] - deepest->axis[2] * projection,
        };
        const float length = length3(radial);
        if (length > kSafeDistance) {
            const float inverse = 1.0f / length;
            for (int axis = 0; axis < 3; ++axis) radial[axis] *= inverse;
        } else {
            // Exactly on the axis: no radial direction exists, so pick one orthogonal to
            // the capsule rather than leaving the vertex buried. Which one does not matter
            // physically, but it has to be deterministic.
            const float reference[3] = { 1.0f, 0.0f, 0.0f };
            const float alternative[3] = { 0.0f, 1.0f, 0.0f };
            const float* pick = std::abs(deepest->axis[0]) < 0.9f ? reference : alternative;
            cross3(deepest->axis, pick, radial);
            const float norm = length3(radial);
            for (int axis = 0; axis < 3; ++axis) radial[axis] /= norm;
        }
        for (int axis = 0; axis < 3; ++axis) point[axis] += -worst * radial[axis];
    }
}

void XpbdSolver::step(const float* guideM, const float* inertialM,
                      const std::vector<Capsule>& capsules, float* outM) {
    stats_ = StepStats{};
    const size_t values = size_t(vertexCount_) * 3;
    std::copy(guideM, guideM + values, outM);
    if (config_.iterations <= 0) {
        // No pass runs and the pin overwrite would copy a value already in place, so the
        // output is `guideM` byte for byte. This is the no-regression gate.
        return;
    }

    // Auxiliary multipliers are fresh per step, exactly as the structural ones are: one
    // call is one substep, so "per call" and "per substep" are the same rule.
    std::fill(multiplier_.begin(), multiplier_.end(), 0.0f);
    std::fill(areaMultiplier_.begin(), areaMultiplier_.end(), 0.0f);
    std::fill(guideMultiplier_.begin(), guideMultiplier_.end(), 0.0f);

    const bool wantArea = triangleCount_ > 0 && config_.areaFloor > 0.0f;
    const bool wantGuide = config_.guideCompliance >= 0.0f;

    // Confidence is computed once, before the loop, from the network's prediction against
    // the inertial one -- not against the evolving position. Against the position it would
    // be zero on the first iteration by construction (the prediction *is* the initial
    // state) and then drift with the solve, which is not a measure of how much to trust
    // the network.
    confidence_.clear();
    if (wantGuide && config_.guideTrustRatio > 0.0f && inertialM != nullptr) {
        confidence_.assign(vertexCount_, 1.0f);
        for (uint32_t vertex = 0; vertex < vertexCount_; ++vertex) {
            const float delta[3] = {
                guideM[3 * vertex] - inertialM[3 * vertex],
                guideM[3 * vertex + 1] - inertialM[3 * vertex + 1],
                guideM[3 * vertex + 2] - inertialM[3 * vertex + 2],
            };
            const float allowed = std::max(config_.guideTrustRatio * minEdge_[vertex], 1.0e-12f);
            // Linear in the excess: a vertex the network wants to move one trust radius is
            // fully trusted, one it wants to move twice that is fully distrusted.
            confidence_[vertex] = std::clamp(2.0f - length3(delta) / allowed, 0.0f, 1.0f);
        }
    }

    for (int iteration = 0; iteration < config_.iterations; ++iteration) {
        // Jacobi: every constraint reads the same positions, so the sweep writes to
        // scratch and the buffers swap. Updating in place would silently become a
        // nondeterministic partial Gauss-Seidel.
        apply_stretch(outM, scratch_.data());
        std::copy(scratch_.begin(), scratch_.end(), outM);
        if (wantArea) {
            apply_area(outM, scratch_.data());
            std::copy(scratch_.begin(), scratch_.end(), outM);
        }
        if (wantGuide) apply_guide(guideM, outM);
        // Contacts last, so the guide cannot pull a vertex back inside the body after the
        // contact resolved it: non-penetration is never traded away for staying near the
        // network's prediction.
        if (config_.collision) apply_contacts(capsules, outM);
    }

    float worst = 0.0f;
    for (uint32_t vertex = 0; vertex < vertexCount_; ++vertex) {
        if (pinned_[vertex]) {
            for (int axis = 0; axis < 3; ++axis) outM[3 * vertex + axis] = guideM[3 * vertex + axis];
            continue;
        }
        float move = 0.0f;
        for (int axis = 0; axis < 3; ++axis) {
            const float d = outM[3 * vertex + axis] - guideM[3 * vertex + axis];
            move += d * d;
        }
        worst = std::max(worst, move);
    }
    stats_.maxCorrectionM = std::sqrt(worst);
}

}  // namespace mlcloth
