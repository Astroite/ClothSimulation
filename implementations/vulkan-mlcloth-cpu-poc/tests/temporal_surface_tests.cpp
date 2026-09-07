#include "demo_surface.h"
#include <iostream>
#include <cassert>
using namespace mlcloth::demo;
static TriangleCollider collider(float angle){
    TriangleCollider b;b.current={{0,0,0},{1,0,0},{0,1,0}};b.triangles={0,1,2};
    const Vec3 axis{std::cos(angle),std::sin(angle),0};MovingCapsule c;c.currentA={0,0,0};c.currentB=axis;c.radius=.1f;
    c.surfaceU={0,0,1};c.surfaceV={std::sin(angle),-std::cos(angle),0};c.hasSurfaceFrame=true;b.capsules.push_back(c);return b;
}
int main(){
    // Cross the former abs(axis.x)==.8 sampling-basis threshold.
    const float critical=std::acos(.8f);auto a=collider(critical-.0001f),b=collider(critical+.0001f);
    auto p=sampleBodySurface(a,b,0),q=sampleBodySurface(b,b,1./30);
    assert(p.ids==q.ids&&p.positions.size()==43);
    for(size_t i=0;i<p.ids.size();i++){assert(length(p.targets[i]-q.positions[i])<1e-7f);assert(length(p.positions[i]-q.positions[i])<.001f);}
    auto bare=p;bare.targets.clear();bare.validate(); // motion cache without a skeleton or lookahead
    auto invalid=bare;invalid.ids[1]=invalid.ids[0];bool rejected=false;try{invalid.validate();}catch(...){rejected=true;}assert(rejected);
    b.capsules[0].hasSurfaceFrame=false;rejected=false;try{sampleBodySurface(b,b,0);}catch(...){rejected=true;}assert(rejected);
    std::cout<<"Stable surface IDs, capsule threshold continuity, optional targets and invalid bindings passed\n";
}
