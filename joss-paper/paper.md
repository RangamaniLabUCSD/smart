---
title: "SMART: Spatial Modelling Algorithms for Reactions and Transport"
tags:
  - Python
  - FEniCS
  - reactions
  - transport
  - biophysics
  - cellular processes
authors:
  - name: Justin Laughlin
    orcid:
    equal-contrib: false
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Jørgen S. Dokken
    orcid:
    affiliation: 2
  - name: Henrik N.T. Finsberg
    orcid:
    affiliation: 3
  - name: Emmet Francis
    orcid:
    affiliation: 1
  - name: Christopher Lee
    orcid:
    affiliation: 1
  - name: Marie E. Rognes
    orcid:
    affiliation: 2
  - name: Padmini Rangamani
    orcid:
    affiliation: 1
    corresponding: true # (This is how to denote the corresponding author)
affiliations:
  - name: Department of Aerospace and Mechanical Engineering, University of California San Diego, La Jolla, CA, USA
    index: 1
  - name: Department of Numerical Analysis and Scientific Computing, Simula Research Laboratory, Oslo, Norway
    index: 2
  - name: Department of Computational Physiology, Simula Research Laboratory, Oslo, Norway
    index: 3
date: April 2023
bibliography:
  - paper.bib
---

# Summary

<!-- A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience. -->

Biological cells respond to stimuli through chains of chemical reactions generally referred to as
*cell signaling pathways*. The propagation of the chemical substances and the cell signalling
pathways can be represented by coupled nonlinear systems of reaction-transport equations within
intracellular, subcellular and extracellular spaces. The geometries of real biological cells are
complicated; subcellular structures such as the endoplasmic reticulum are highly curved and
tortuous. *Spatial Modelling Algorithms for Reactions and Transport* (SMART) is a high-performance
finite-element-based simulation package for describing, modelling and simulating spatially-varying
reaction-transport processes with specific features targeting signaling pathways in biological cell
geometries. SMART is based on the FEniCS finite element library, provides a symbolic representation
framework for specifying reaction pathways, supports general 2D and 3D geometry specifications, and
supports parallelism via MPI.

# Statement of need

<!-- A Statement of need section that clearly illustrates the research purpose of the software and places it in the context of related work. -->

The traditional approach to modelling cell signalling pathways is to assume that the substances are
well-mixed within the cell body or a subcellular compartment. In this case, the governing
biophysical equations reduce to ordinary differential equations, for which a plethora of simulation
tools exist such as VCell [@Schaff:1997] and COPASI [@Hoops:2006]. This approach has successfully
recapitulated many cell-wide signaling events, from calcium elevations in neurons to models of cell
mechanotransduction. However, such models obviously neglect many spatial aspects of cell signaling,
which are crucially important on short timescales or for slower diffusing species. Recently, there
has been increased interest in spatiotemporal modeling of cell signaling, but

* Describe in 1-2 sentences which frameworks that exist in addition to SMART
* State in 1-2 sentences why these are insufficient
* Describe in 1-2 sentence key SMART features that addresses these insufficiencies.
* Include citations, see how to format citations in text.

Other software tools currently used to model signaling networks in cells either primarily focus on
assembling non-spatial models or lack sufficient flexibility to model sufficiently complex
geometries (e.g. are restricted to Cartesian meshes [@Cowan:2012]). On the other hand, there exist
many mature platforms that use the finite element method to solve various PDEs (COMSOL, ABAQUS);
however, these generally lack the flexibility to adapt to the highly nonlinear systems of mixed
dimensional PDEs present in signaling networks in cells. SMART leverages state-of-the-art finite
element software (FEniCS) [@Logg:2010] which is compatible with highly flexible meshing software
such as Gmsh [@Geuzaine:2009] or the newly developed GAMer 2 [@Lee:2020], allowing users to solve
highly nonlinear systems of PDEs within complex cellular geometries.


# Scientific impact and examples of SMART use

* Detail one existing and some upcoming use cases for SMART.
* Include 1-2 figures, and 2-3 references.

SMART offers the unique opportunity to model signaling networks spatially in realistic cell
geometries; for instance, using recent electron micrographs of ER geometry in Purkinje neurons, we
were able to predict the emergent calcium dynamics within realistic cell volumes (Fig 1). …[mention
another example?] In the near future, SMART can be used to solve coupled mechano-chemical systems;
for instance, we plan to explore coupling between calcium release and contraction within single
skeletal muscle fibers.

<!-- Figures can be included like this: -->
<!-- ![Caption for example figure.\label{fig:example}](figure.png) -->
<!-- and referenced from text using \autoref{fig:example}. -->


<!-- # Citations -->

<!-- Citations to entries in paper.bib should be in -->
<!-- [rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html) -->
<!-- format. -->

<!-- If you want to cite a software repository URL (e.g. something on GitHub without a preferred -->
<!-- citation) then you can do it with the example BibTeX entry below for @fidgit. -->

<!-- For a quick reference, the following citation commands can be used: -->
<!-- - `@author:2001`  ->  "Author et al. (2001)" -->
<!-- - `[@author:2001]` -> "(Author et al., 2001)" -->
<!-- - `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)" -->

<!-- # Figures -->


<!-- Figure sizes can be customized by adding an optional second parameter: -->
<!-- ![Caption for example figure.](figure.png){ width=20% } -->

# Acknowledgements

Acknowledgments.

# References

{#refs}
