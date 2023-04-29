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
  - name: Jørgen Dokken
    orcid: 
    affiliation: 2 
  - name: Henrik Finsberg
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

Biological cells predominantly respond to stimulus by chains of
chemical reactions in a process called *cell signaling* and by way of
*cell signaling pathways*. The propagation of the chemical substances
and the cell signalling pathways can be represented by coupled
nonlinear systems of reaction-transport equations within
intracellular, subcellular and extracellular spaces. The geometries of
real biological cells, subcellular structures such as the endoplasmic
reticulum, as well as the tortuous and narrow extracellular space are
complicated, with highly lobed and segmented morphologies. *Spatial
Modelling Algorithms for Reactions and Transport* (SMART) is a
high-performance finite-element-based simulation package for
describing, modelling and simulating spatially-varying
reaction-transport processes with specific features targeting
signalling pathways in biological cell geometries. SMART is based on
the FEniCS finite element library, provides a symbolic representation
framework for specifying reaction pathways, supports general 2D and 3D
geometry specifications, and supports parallelism via MPI. Its target
audience is ...

# Statement of need

<!-- A Statement of need section that clearly illustrates the research purpose of the software and places it in the context of related work. -->

The traditional approach to modelling cell signalling pathways is to
assume that the substances are well-mixed within the cell body or a
subcellular compartment. In this case, the governing biophysical
equations reduce to ordinary differential equations, for which a
plethora of simulation tools exist [@xxx:20XX; @xxx:20XX]. This
approach has successfully recapitulated many cell-wide signaling
events, from calcium elevations in neurons to models of cell
mechanotransduction. However, such models obviously neglect many
spatial aspects of cell signaling, which are crucially important on
short timescales or for slower diffusing species. Recently, there has
been increased interest in spatiotemporal modeling of cell signaling. 

 
* Describe in 1-2 sentences which frameworks that exist in addition to SMART
* State in 1-2 sentences why these are insufficient
* Describe in 1-2 sentence key SMART features that addresses these insufficiencies.
* Include citations, see how to format citations in text.

# Scientific impact and examples of SMART use

* Detail one existing and some upcoming use cases for SMART. 
* Include 1-2 figures, and 2-3 references.

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
