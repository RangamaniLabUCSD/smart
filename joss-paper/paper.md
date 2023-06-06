---
title: "SMART: Spatial Modeling Algorithms for Reactions and Transport"
tags:
  - Python
  - FEniCS
  - reactions
  - transport
  - biophysics
  - cellular processes
authors:
  - name: Justin G. Laughlin
    orcid: 0000-0001-8229-9760
    equal-contrib: false
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Jørgen S. Dokken
    orcid:
    affiliation: 2
  - name: Henrik N.T. Finsberg
    orcid: 0000-0003-3766-2393
    affiliation: 3
  - name: Emmet Francis
    orcid:
    affiliation: 1
  - name: Christopher T. Lee
    orcid: 0000-0002-0670-2308
    affiliation: 1
  - name: Marie E. Rognes
    orcid: 0000-0002-6872-3710
    affiliation: 2
  - name: Padmini Rangamani
    orcid: 0000-0001-5953-4347
    affiliation: 1
    corresponding: true # (This is how to denote the corresponding author)
affiliations:
  - name: Department of Aerospace and Mechanical Engineering, University of California San Diego, La Jolla, CA, USA
    index: 1
  - name: Department of Numerical Analysis and Scientific Computing, Simula Research Laboratory, Oslo, Norway
    index: 2
  - name: Department of Computational Physiology, Simula Research Laboratory, Oslo, Norway
    index: 3
date: June 2023
bibliography:
  - paper.bib
---

# Summary

<!-- A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience. -->

Recent advances in microscopy and 3D reconstruction methods allow for describing biological cell morphology at unprecedented detail,
including the highly irregular geometries of intracellular subcompartments such as membrane-bound organelles.
These geometries are now compatible with predictive modeling of cellular function. Biological cells respond to stimuli
through chains of chemical reactions generally referred to as *cell signaling pathways*.
The propagation and reaction of chemical substances in cell signaling pathways can be represented by coupled nonlinear
systems of reaction-transport equations.
These reaction pathways include numerous chemical species that react across boundaries or interfaces
(*e.g.* the cell membrane and membranes of organelles within the cell) and domains
(*e.g.* the bulk cell volume and the interior of organelles).
Such systems of multi-dimensional partial differential equations (PDEs) are notoriously difficult to solve
because of their high dimensionality, non-linearities, strong coupling, stiffness, and potential instabilities.
In this work, we describe *Spatial Modeling Algorithms for Reactions and Transport* (SMART),
a high-performance finite-element-based simulation package for model specification and numerical simulation of spatially-varying reaction-transport processes.
SMART is based on the FEniCS finite element library, provides a symbolic representation
framework for specifying reaction pathways, and supports large and irregular cell geometries in 2D and 3D.

# Statement of need

<!-- A Statement of need section that clearly illustrates the research purpose of the software and places it in the context of related work. -->

SMART has been designed to fulfill the need for an open-source software capable of modeling cell signaling pathways within complicated cell geometries,
including reactions and transport between different subcellular surfaces and volumes.
In SMART, the user specifies *species, reactions, compartments, and parameters* to define a high-level model representation.
This framework uses a similar convention to Systems Biology Markup Language (SBML, [@Schaff:2023]),
making the software approachable for a wider user base.
SMART provides features for converting the model representation into appropriate coupled systems
of ordinary differential equations (ODEs) and PDEs,
and for solving these efficiently using finite element and finite difference discretizations.

<!-- * Describe in 1-2 sentences which frameworks that exist in addition to SMART
* State in 1-2 sentences why these are insufficient
* Describe in 1-2 sentence key SMART features that addresses these insufficiencies.
* Include citations, see how to format citations in text. -->

SMART has been designed for use by computational biologists and biophysicists.
SMART leverages state-of-the-art finite element software (FEniCS) [@Logg:2012; @Alnæs:2015]
which is compatible with a variety of meshing software such as Gmsh [@Geuzaine:2009]
or the newly developed GAMer 2 [@Lee:2020], allowing users to solve
highly nonlinear systems of PDEs within complex cellular geometries.
Moreover, the design of SMART as a FEniCS-based package allows for ease of extension and integration
with additional physics, enabling *e.g.* coupled simulations of cell signaling and mechanics or electrophysiology.
SMART complements several existing software tools that are used to assemble and solve equations
describing cell signaling networks such as VCell [@Cowan:2012; @Schaff:1997], COPASI [@Hoops:2006], and MCell [@Kerr:2008].


# Examples of SMART use

<!-- * Detail one existing and some upcoming use cases for SMART.
* Include 1-2 figures, and 2-3 references. -->

SMART offers unique opportunities to examine the behavior of signaling networks
in realistic cell geometries. As a proof of concept, we previously used SMART to model
a coupled volume-surface reaction-diffusion system on a mesh of a dendritic spine generated by
our GAMer 2 software (\autoref{fig:fig1}, [@Lee:2020]).
More recently, we implemented a detailed model of neuron calcium dynamics in SMART (\autoref{fig:fig2}).
This model describes $\mathrm{IP_3R}$- and ryanodine receptor (RyR)-mediated
calcium release following stimulation by neurotransmitters.
These SMART simulations recapitulate the complex dynamics of calcium-induced
calcium release from the endoplasmic reticulum and predict strong
spatial gradients of calcium near regions of calcium release (\autoref{fig:fig2}C).

<!-- Figures can be included like this: -->
<!-- ![Caption for example figure.\label{fig:example}](figure.png) -->
<!-- and referenced from text using \autoref{fig:example}. -->

![Simulation of a surface-volume reaction in a realistic dendritic spine geometry using SMART. A) Diagram of the chosen surface-volume reaction, in which cytosolic species, A, reacts with a species in the membrane, X, to produce a new membrane species, B (originally described in [@Rangamani:2013]). Note that this is the same reaction used in Example 2 of the SMART demos. B) Geometry-preserving mesh of a neuronal dendritic spine attached to a portion of the dendritic shaft, constructed from electron microscopy data of a mouse neuron using GAMer 2. The mesh contains two domains - the surface, $\Gamma_{PM}$, which represents the plasma membrane, and the inner volume, $\Omega_{Cyto}$, which represents the cytosol. C) Concentration of product B on the plasma membrane at $t=1.0$ s, with the diffusion coefficient of species A ($D_A$) set to 10 μ$\mathrm{m^2}$/s. D) Range of concentrations of species B over time for the simulation shown in (C), where the solid lines denote the minimum and maximum concentrations at each time point and the dotted line indicates the average concentration. This figure was adapated from Fig 10 in [@Lee:2020]; additional parameters and details given in the original paper. \label{fig:fig1}](JOSS_Fig1.png){width=80%}

![Model of calcium dynamics in a neuron using SMART (implemented in Example 6). A) Diagram of the calcium signaling network in the main body (soma) of a neuron. $\mathrm{IP_3}$ production at the plasma membrane (PM) triggers the opening of $\mathrm{IP_3R}$ calcium channels in the endoplasmic reticulum (ER) membrane, leading to calcium elevations in the cytosol. In parallel, calcium entry through voltage-gated calcium channels (VGCCs) and calcium release from the ER through ryanodine receptors (RyRs) also increase cytosolic calcium levels, while calcium export through the plasma membrane ATPase (PMCA) and sodium-calcium exchanger (NCX) and pumping of calcium back into the ER via the sarco-endoplasmic reticulum ATPase (SERCA) counteract these calcium increases. Calcium rapidly binds to other proteins in the cytosol such as parvalbumin (PV) and calbindin-D28k (CD28k), which effectively act as calcium buffers. For the mathematical details of this model, see Example 6 in the SMART demos. The mesh depicted on the right shows the "sphere-in-a-sphere" geometry tested in Example 6, in which the inner sphere corresponds to the ER and the outer region corresponds to the cytosol in a portion of the neuron soma. B) Plots of the time-dependent activation functions, corresponding to calcium entry through VGCCs (upper plot) and $\mathrm{IP_3}$ production at the plasma membrane. Patterns of calcium influx were derived from those used in [@Doi:2005], and $\mathrm{IP_3}$ production was fit to expected values from simulating a larger signaling network of glutamate-dependent $\mathrm{IP_3}$ production. C) Cytosolic and ER calcium concentrations plotted over the mesh at indicated time points. After the final "simple spike" of calcium at $t=0.05$ s, $\mathrm{IP_3}$ production slowly leads to a small amount of calcium release from the ER. However, once the "complex spike" occurs at $t=0.1$ s, a larger amount of calcium is released from the ER, manifesting as a sharp local gradient around the ER that is visible at $t=0.111$ s. D) Plots of the average cytosolic calcium (upper plot) and average ER calcium (lower plot) over time for the simulation shown in (C). Note that the plots shown in (B) and (D) can be automatically generated by running Example 6 in the SMART demos. \label{fig:fig2}](JOSS_Fig2.png)


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

The authors would like to acknowledge contributions from Yuan Gao and William Xu during the early development of SMART.

This material is based upon work supported by the National Science Foundation under Grant #EEC-2127509 to the American Society for Engineering Education. MER acknowledges support and funding from the Research Council of Norway (RCN) via FRIPRO grant agreement #324239 (EMIx), and the U.S.-Norway Fulbright Foundation for Educational Exchange.

# References
