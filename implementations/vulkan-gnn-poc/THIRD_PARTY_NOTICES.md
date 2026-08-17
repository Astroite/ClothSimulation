# Third-party notices

## Sascha Willems Vulkan examples

The bootstrap checkout and the `gnncloth` sample's application structure,
rendering setup, ping-pong buffers, compute/graphics queue ownership transfer,
and semaphore flow are derived from
[SaschaWillems/Vulkan](https://github.com/SaschaWillems/Vulkan), pinned in
`upstream.lock.json`.

MIT License

Copyright (c) 2016-2026, Sascha Willems

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Architecture references

- PyTorch Geometric's public two-layer GCN example was used as a structural
  reference only; no PyG code or runtime is included.
- Khronos Vulkan documentation informed the compute-layer and cross-queue
  synchronization design; documentation text is not redistributed.

## HOOD Fine15

The real-character path consumes the official `fine15.pth` checkpoint from
[dolorousrtur/hood](https://github.com/dolorousrtur/hood), pinned to commit
`9bc1076195979ac6c027fdd729c6e960cad62f2a`. The downloader verifies checkpoint
SHA-256 `bc92f1fb9a0ca1c9e476ad3981c3e4453bd66519ef16e6f2d6a52305c2aa13cb`.
The raw checkpoint and generated `VHOOD` file are local cache artifacts and are
not committed by this PoC. HOOD is distributed under the MIT License; its
copyright and license remain with its authors.
