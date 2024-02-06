# Isosurface Mesh Extraction Benchmarking Tool (WIP)

## About
This tool is being developed for a seminar paper on isosurface mesh extraction algorithms. 
The aim of the paper is to go over some of the most popular and novel algorithms and benchmark them on a large dataset. 
The tool allows for automated running of the datasets through the selected extraction techniques and measuring performance, accuracy and quality of the extracted meshes.

Inspired by NVIDIAs [Flexicubes paper](https://research.nvidia.com/labs/toronto-ai/flexicubes/)

## Paper draft
At the time of writing this, the paper is not finished yet, however it already contains an overview of the benchmarking process, including descriptions of datasets and performance, quality and accuracy metrics used. 
All content of this draft is subject to change by I am attching it here for a more detailed description of the current approach.

[CG_Seminar_Jakub_Nawrocki_Isosurface_Mesh_Extraction.pdf](https://github.com/ldkuba/cg-seminar-2023/files/14183982/CG_Seminar_Jakub_Nawrocki_Isosurface_Mesh_Extraction.pdf)

## Extraction Algorithms
__Note about implementations:__ Currently none of the implementations are my own. Each of the selected techniques is benchmarked using an open source implementation. 
More details about the sources of these implementations can be found in the paper draft.

### Implemented techniques
- Marching Cubes
- Reach For The Spheres
- Flexicubes
### Currently planned
- Dual Neural Contouring
- Deep Marching Tetrahedra

## Usage
The tool is not yet ready for distribution. I will update this section once it is.
