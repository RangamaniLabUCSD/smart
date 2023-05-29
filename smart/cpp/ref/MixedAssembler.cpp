// Copyright (C) 2007-2015 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Cecile Daversin-Catty 2017

#include <algorithm>
#include <dolfin/log/log.h>
#include <dolfin/log/Progress.h>
#include <dolfin/common/ArrayView.h>
#include <dolfin/common/Timer.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/la/GenericTensor.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/SubDomain.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/FunctionSpace.h>
#include "GenericDofMap.h"
#include "Form.h"
#include "UFC.h"
#include "FiniteElement.h"
#include "AssemblerBase.h"
#include "MixedAssembler.h"

#include <dolfin/mesh/MultiMesh.h>
#include <dolfin/la/GenericMatrix.h>

using namespace dolfin;

//----------------------------------------------------------------------------
void MixedAssembler::assemble(GenericTensor& A, const Form& a)
{
  // All assembler functions above end up calling this function, which
  // in turn calls the assembler functions below to assemble over
  // cells, exterior and interior facets.

  // Get cell domains
  std::shared_ptr<const MeshFunction<std::size_t>>
    cell_domains = a.cell_domains();

  // Get exterior facet domains
  std::shared_ptr<const MeshFunction<std::size_t>> exterior_facet_domains
      = a.exterior_facet_domains();

  // Get interior facet domains
  std::shared_ptr<const MeshFunction<std::size_t>> interior_facet_domains
      = a.interior_facet_domains();

  // Get vertex domains
  std::shared_ptr<const MeshFunction<std::size_t>> vertex_domains
    = a.vertex_domains();

  // Check form
  AssemblerBase::check(a);

  // Create data structure for local assembly data
  UFC ufc(a);

  // Update off-process coefficients
  const std::vector<std::shared_ptr<const GenericFunction>>
    coefficients = a.coefficients();

  // Initialize global tensor
  init_global_tensor(A, a);

  // Assemble over cells
  assemble_cells(A, a, ufc, cell_domains, NULL);

  // Assemble over exterior facets
  assemble_exterior_facets(A, a, ufc, exterior_facet_domains, NULL);

  // Assemble over interior facets
  assemble_interior_facets(A, a, ufc, interior_facet_domains,
                           cell_domains, NULL);

  // Assemble over vertices
  assemble_vertices(A, a, ufc, vertex_domains);

  // Finalize assembly of global tensor
  if (finalize_tensor)
    A.apply("add");
}
//-----------------------------------------------------------------------------
void MixedAssembler::assemble_cells(
  GenericTensor& A,
  const Form& a,
  UFC& ufc,
  std::shared_ptr<const MeshFunction<std::size_t>> domains,
  std::vector<double>* values)
{
  // Skip assembly if there are no cell integrals
  if (!ufc.form.has_cell_integrals())
    return;

  // Set timer
  Timer timer("[MixedAssembler] Assemble cells");

  // Extract mesh
  dolfin_assert(a.mesh());
  const Mesh& mesh = *(a.mesh());

  // Form rank
  const std::size_t form_rank = ufc.form.rank();

  // Check if form is a functional
  const bool is_cell_functional = (values && form_rank == 0) ? true : false;

  // Iterations depending on rank or coefficients when rank is zero
  std::size_t rn = form_rank;
  if (form_rank == 0)
    rn = ufc.form.num_coefficients();

  // Collect pointers to dof maps / id of the involved meshes
  std::vector<const GenericDofMap*> dofmaps;
  std::vector<unsigned> mesh_id(rn);

  // Meshes and dofmaps from Trial and Test functions if rank>0
  for (std::size_t i = 0; i < form_rank; ++i)
  {
    mesh_id[i] = a.function_space(i)->mesh()->id();
    dofmaps.push_back(a.function_space(i)->dofmap().get());
  }
  // Meshes from coefficients otherwise
  if (form_rank == 0)
  {
    for (std::size_t i=0; i<a.coefficients().size(); ++i)
    {
      mesh_id[i] = a.coefficients()[i]->function_space()->mesh()->id();
      dofmaps.push_back(a.coefficients()[i]->function_space()->dofmap().get());
    }
  }

  // Vector to hold dof map for a cell
  std::vector<ArrayView<const dolfin::la_index>> dofs(rn);

  // Cell integral
  ufc::cell_integral* integral = ufc.default_cell_integral.get();

  // Check whether integral is domain-dependent
  bool use_domains = domains && !domains->empty();

  // Assemble over cells
  ufc::cell ufc_cell;
  std::vector<double> coordinate_dofs;
  Progress p(AssemblerBase::progress_message(A.rank(), "cells"),
             mesh.num_cells());

  auto mapping_map = mesh.topology().mapping();
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Get integral for sub domain (if any)
    if (use_domains)
      integral = ufc.get_cell_integral((*domains)[*cell]);

    // Skip if no integral on current domain
    if (!integral)
      continue;

    // Check that cell is not a ghost
    dolfin_assert(!cell->is_ghost());

    // Update to current cell
    cell->get_cell_data(ufc_cell);
    cell->get_coordinate_dofs(coordinate_dofs);

    ufc.update(*cell, coordinate_dofs, ufc_cell,
               integral->enabled_coefficients());

    // Get local-to-global dof maps for cell
    bool empty_dofmap = false;
    std::vector<std::vector<std::size_t>> cell_index(rn);
    // Mixed-dimensional
    std::vector<std::size_t> codim(rn);
    std::vector<int> local_facets;

    for (size_t i = 0; i < rn; ++i)
    {
      cell_index[i].push_back(cell->index());

      if (mesh_id[i] != mesh.id() && mapping_map[mesh_id[i]])
      {
	auto mapping = mapping_map[mesh_id[i]];
	dolfin_assert(mapping->mesh()->id() == mesh_id[i]);

	codim[i] = mapping->mesh()->topology().dim() - mesh.topology().dim();
	if(codim[i] == 0)
	  cell_index[i][0] = mapping->cell_map()[cell->index()];
	else if(codim[i] == 1) // 2D-1D or 3D-2D (cells - facets relationships)
	{
	  const std::size_t D = mapping->mesh()->topology().dim();
	  mapping->mesh()->init(D);
	  mapping->mesh()->init(D - 1, D);

	  Facet mesh_facet(*(mapping->mesh()), mapping->cell_map()[cell->index()]);
	  for(std::size_t j=0; j<mesh_facet.num_entities(D);j++)
	  {
	    Cell mesh_cell(*(mapping->mesh()), mesh_facet.entities(D)[j]);
	    local_facets.push_back(mesh_cell.index(mesh_facet));

	    if(j==0)
	      cell_index[i][0] = mesh_cell.index();
	    else
	      cell_index[i].push_back(mesh_cell.index());
	  }
	}
	else if(codim[i] == 2) // 3D-1D (cells - edges relationships)
	  std::cout << "[MixedAssembler] codim 2 - Not implemented (yet)" << std::endl;
      }
    }

    if (empty_dofmap)
      continue;

    if(local_facets.empty()) // Not mixed-dimensional - Standard case
    {
      integral->tabulate_tensor(ufc.A.data(),
				ufc.w(),
				coordinate_dofs.data(),
				ufc_cell.orientation);

      for(std::size_t i=0; i<rn; ++i)
      {
	auto dmap = dofmaps[i]->cell_dofs(cell_index[i][0]);
	dofs[i].set(dmap.size(), dmap.data());
      }

      if (is_cell_functional)
	(*values)[cell->index()] = ufc.A[0];
      else
	A.add_local(ufc.A.data(), dofs);
    }
    else // Mixed-dimensional integrals
    {
      for(std::size_t j=0; j<local_facets.size(); ++j)
      {
	integral->tabulate_tensor(ufc.A.data(),
				  ufc.w(),
				  coordinate_dofs.data(),
				  ufc_cell.orientation,
				  local_facets[j]); // local index of the lower dim entity involved


        for(std::size_t i=0; i<rn; ++i)
	{
          std::size_t jidx = (cell_index[i].size() > 1) ? j:0;

          auto dmap = dofmaps[i]->cell_dofs(cell_index[i][jidx]);

          // If there is more than one higher-dimensional contrib, do not consider the same dof twice
          // #FIXME : This could be more elegant
	  for(std::size_t dof=0; jidx != 0 && dof<dofs[i].size(); ++dof)
	  {
	    for(int rm = 0; rm<dmap.size(); rm++)
	    {
              if(form_rank > 1 && dmap[rm] == dofs[i][dof]) // This dof (index=rm) has already been set
	      {
		if(i == 0)
		{
		  for(std::size_t col=0; col<dofmaps[1]->max_element_dofs(); ++col)
		    ufc.A[rm*dofmaps[1]->max_element_dofs() + col] = 0.0;
		}
	        if(i == 1)
		{
		  for(std::size_t row=0; row<dofmaps[0]->max_element_dofs(); ++row)
		    ufc.A[row*dofmaps[1]->max_element_dofs() + rm] = 0.0;
		}
	      }
	    }
	  }

	  dofs[i].set(dmap.size(), dmap.data());
	}

        // TO CHECK/IMPROVE : If we have contributions from adjacent cells from different meshes
        if(local_facets.size() > 1 && cell_index[0].size() <= 1 && cell_index[rn - 1].size() <= 1)
          for(auto& m: ufc.A)
            m/=local_facets.size();

	if (is_cell_functional)
	  (*values)[cell->index()] = ufc.A[0];
	else
	  A.add_local(ufc.A.data(), dofs);
      }
    }
    p++;
  }
  timer.stop();
}
//-----------------------------------------------------------------------------
void MixedAssembler::assemble_exterior_facets(
  GenericTensor& A,
  const Form& a,
  UFC& ufc,
  std::shared_ptr<const MeshFunction<std::size_t>> domains,
  std::vector<double>* values)
{
  // Skip assembly if there are no exterior facet integrals
  if (!ufc.form.has_exterior_facet_integrals())
    return;

  // Set timer
  Timer timer("Assemble exterior facets");

  // Extract mesh
  dolfin_assert(a.mesh());
  const Mesh& mesh = *(a.mesh());

  // Form rank
  const std::size_t form_rank = ufc.form.rank();

  // Collect pointers to dof maps
  std::vector<const GenericDofMap*> dofmaps;
  for (std::size_t i = 0; i < form_rank; ++i)
    dofmaps.push_back(a.function_space(i)->dofmap().get());

  // Vector to hold dof map for a cell
  std::vector<ArrayView<const dolfin::la_index>> dofs(form_rank);

  // Exterior facet integral
  const ufc::exterior_facet_integral* integral
    = ufc.default_exterior_facet_integral.get();

  // Check whether integral is domain-dependent
  bool use_domains = domains && !domains->empty();

  // Compute facets and facet - cell connectivity if not already computed
  const std::size_t D = mesh.topology().dim();
  mesh.init(D - 1);
  mesh.init(D - 1, D);
  dolfin_assert(mesh.ordered());

  // Assemble over exterior facets (the cells of the boundary)
  ufc::cell ufc_cell;
  std::vector<double> coordinate_dofs;
  Progress p(AssemblerBase::progress_message(A.rank(), "exterior facets"),
             mesh.num_facets());
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    // Only consider exterior facets
    if (!facet->exterior())
    {
      p++;
      continue;
    }

    // Get integral for sub domain (if any)
    if (use_domains)
      integral = ufc.get_exterior_facet_integral((*domains)[*facet]);

    // Skip integral if zero
    if (!integral)
      continue;

    // Get mesh cell to which mesh facet belongs (pick first, there is
    // only one)
    dolfin_assert(facet->num_entities(D) == 1);
    Cell mesh_cell(mesh, facet->entities(D)[0]);

    // Check that cell is not a ghost
    dolfin_assert(!mesh_cell.is_ghost());

    // Get local index of facet with respect to the cell
    const std::size_t local_facet = mesh_cell.index(*facet);

    // Update UFC cell
    mesh_cell.get_cell_data(ufc_cell, local_facet);
    mesh_cell.get_coordinate_dofs(coordinate_dofs);

    // Update UFC object
    ufc.update(mesh_cell, coordinate_dofs, ufc_cell,
               integral->enabled_coefficients());

    // Get local-to-global dof maps for cell
    for (std::size_t i = 0; i < form_rank; ++i)
    {
      auto dmap = dofmaps[i]->cell_dofs(mesh_cell.index());
      dofs[i].set(dmap.size(), dmap.data());
    }

    // Tabulate exterior facet tensor
    integral->tabulate_tensor(ufc.A.data(),
                              ufc.w(),
                              coordinate_dofs.data(),
                              local_facet,
                              ufc_cell.orientation);

    // Add entries to global tensor
    A.add_local(ufc.A.data(), dofs);

    p++;
  }
}
//-----------------------------------------------------------------------------
void MixedAssembler::assemble_interior_facets(
  GenericTensor& A,
  const Form& a,
  UFC& ufc,
  std::shared_ptr<const MeshFunction<std::size_t>> domains,
  std::shared_ptr<const MeshFunction<std::size_t>> cell_domains,
  std::vector<double>* values)
{
  // Skip assembly if there are no interior facet integrals
  if (!ufc.form.has_interior_facet_integrals())
    return;

  // Set timer
  Timer timer("Assemble interior facets");

  // Extract mesh and coefficients
  dolfin_assert(a.mesh());
  const Mesh& mesh = *(a.mesh());

  // Sanity check of ghost mode (proper check in AssemblerBase::check)
  dolfin_assert(mesh.ghost_mode() == "shared_vertex"
                || mesh.ghost_mode() == "shared_facet"
                || MPI::size(mesh.mpi_comm()) == 1);

  // MPI rank
  const int my_mpi_rank = MPI::rank(mesh.mpi_comm());

  // Form rank
  const std::size_t form_rank = ufc.form.rank();

  // Collect pointers to dof maps
  std::vector<const GenericDofMap*> dofmaps;
  for (std::size_t i = 0; i < form_rank; ++i)
    dofmaps.push_back(a.function_space(i)->dofmap().get());

  // Vector to hold dofs for cells, and a vector holding pointers to same
  std::vector<std::vector<dolfin::la_index>> macro_dofs(form_rank);
  std::vector<ArrayView<const dolfin::la_index>> macro_dof_ptrs(form_rank);

  // Interior facet integral
  const ufc::interior_facet_integral* integral
    = ufc.default_interior_facet_integral.get();

  // Check whether integral is domain-dependent
  bool use_domains = domains && !domains->empty();
  bool use_cell_domains = cell_domains && !cell_domains->empty();

  // Compute facets and facet - cell connectivity if not already computed
  const std::size_t D = mesh.topology().dim();
  mesh.init(D - 1);
  mesh.init(D - 1, D);
  dolfin_assert(mesh.ordered());

  // Assemble over interior facets (the facets of the mesh)
  ufc::cell ufc_cell[2];
  std::vector<double> coordinate_dofs[2];
  Progress p(AssemblerBase::progress_message(A.rank(), "interior facets"),
             mesh.num_facets());
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    if (facet->num_entities(D) == 1)
      continue;

    // Check that facet is not a ghost
    dolfin_assert(!facet->is_ghost());

    // Get integral for sub domain (if any)
    if (use_domains)
      integral = ufc.get_interior_facet_integral((*domains)[*facet]);

    // Skip integral if zero
    if (!integral)
      continue;

    // Get cells incident with facet (which is 0 and 1 here is arbitrary)
    dolfin_assert(facet->num_entities(D) == 2);
    std::size_t cell_index_plus = facet->entities(D)[0];
    std::size_t cell_index_minus = facet->entities(D)[1];

    if (use_cell_domains && (*cell_domains)[cell_index_plus]
        < (*cell_domains)[cell_index_minus])
    {
      std::swap(cell_index_plus, cell_index_minus);
    }

    // The convention '+' = 0, '-' = 1 is from ffc
    const Cell cell0(mesh, cell_index_plus);
    const Cell cell1(mesh, cell_index_minus);

    // Get local index of facet with respect to each cell
    std::size_t local_facet0 = cell0.index(*facet);
    std::size_t local_facet1 = cell1.index(*facet);

    // Update to current pair of cells
    cell0.get_cell_data(ufc_cell[0], local_facet0);
    cell0.get_coordinate_dofs(coordinate_dofs[0]);
    cell1.get_cell_data(ufc_cell[1], local_facet1);
    cell1.get_coordinate_dofs(coordinate_dofs[1]);

    ufc.update(cell0, coordinate_dofs[0], ufc_cell[0],
               cell1, coordinate_dofs[1], ufc_cell[1],
               integral->enabled_coefficients());

    // Tabulate dofs for each dimension on macro element
    for (std::size_t i = 0; i < form_rank; i++)
    {
      // Get dofs for each cell
      auto cell_dofs0 = dofmaps[i]->cell_dofs(cell0.index());
      auto cell_dofs1 = dofmaps[i]->cell_dofs(cell1.index());

      // Create space in macro dof vector
      macro_dofs[i].resize(cell_dofs0.size() + cell_dofs1.size());

      // Copy cell dofs into macro dof vector
      std::copy(cell_dofs0.data(), cell_dofs0.data() + cell_dofs0.size(),
                macro_dofs[i].begin());
      std::copy(cell_dofs1.data(), cell_dofs1.data() + cell_dofs1.size(),
                macro_dofs[i].begin() + cell_dofs0.size());
      macro_dof_ptrs[i].set(macro_dofs[i]);
    }

    // Tabulate interior facet tensor on macro element
    integral->tabulate_tensor(ufc.macro_A.data(),
                              ufc.macro_w(),
                              coordinate_dofs[0].data(),
                              coordinate_dofs[1].data(),
                              local_facet0,
                              local_facet1,
                              ufc_cell[0].orientation,
                              ufc_cell[1].orientation);

    if (cell0.is_ghost() != cell1.is_ghost())
    {
      int ghost_rank = -1;
      if (cell0.is_ghost())
        ghost_rank = cell0.owner();
      else
        ghost_rank = cell1.owner();

      dolfin_assert(my_mpi_rank != ghost_rank);
      dolfin_assert(ghost_rank != -1);
      if (ghost_rank < my_mpi_rank)
        continue;
    }

    // Add entries to global tensor
    A.add_local(ufc.macro_A.data(), macro_dof_ptrs);

    p++;
  }
}
//-----------------------------------------------------------------------------
void MixedAssembler::assemble_vertices(
  GenericTensor& A,
  const Form& a,
  UFC& ufc,
  std::shared_ptr<const MeshFunction<std::size_t>> domains)
{
  // Skip assembly if there are no point integrals
  if (!ufc.form.has_vertex_integrals())
    return;

  // Set timer
  Timer timer("Assemble vertices");

  // Extract mesh
  dolfin_assert(a.mesh());
  const Mesh& mesh = *(a.mesh());

  // Compute cell and vertex - cell connectivity if not already
  // computed
  const std::size_t D = mesh.topology().dim();
  mesh.init(0);
  mesh.init(0, D);
  dolfin_assert(mesh.ordered());

  // Logics for shared vertices
  const bool has_shared_vertices = mesh.topology().have_shared_entities(0);
  const std::map<std::int32_t, std::set<unsigned int>>&
    shared_vertices = mesh.topology().shared_entities(0);

  // Form rank
  const std::size_t form_rank = ufc.form.rank();

  // Collect pointers to dof maps
  std::vector<const GenericDofMap*> dofmaps(form_rank);

  // Create a vector for storying local to local map for vertex entity
  // dofs
  std::vector<std::vector<std::size_t>> local_to_local_dofs(form_rank);

  // Create a values vector to be used to fan out local tabulated
  // values to the global tensor
  std::vector<double> local_values(1);

  // Vector to hold local dof map for a vertex
  std::vector<std::vector<dolfin::la_index>> global_dofs(form_rank);
  std::vector<ArrayView<const dolfin::la_index>> global_dofs_p(form_rank);
  std::vector<dolfin::la_index> local_dof_size(form_rank);
  for (std::size_t i = 0; i < form_rank; ++i)
  {
    dofmaps[i] = a.function_space(i)->dofmap().get();

    // Check that the test and trial space as dofs on the vertices
    if (dofmaps[i]->num_entity_dofs(0) == 0)
    {
      dolfin_error("MixedAssembler.cpp",
                   "assemble form over vertices",
                   "Expecting test and trial spaces to have dofs on "\
                   "vertices for point integrals");
    }

    // Check that the test and trial spaces do not have dofs other
    // than on vertices
    for (std::size_t j = 1; j <= D; j++)
    {
      if (dofmaps[i]->num_entity_dofs(j)!=0)
      {
        dolfin_error("MixedAssembler.cpp",
                     "assemble form over vertices",
                     "Expecting test and trial spaces to only have dofs on " \
                     "vertices for point integrals");
      }
    }

    // Resize local values so it can hold dofs on one vertex
    local_values.resize(local_values.size()*dofmaps[i]->num_entity_dofs(0));

    // Resize local to local map according to the number of vertex
    // entities dofs
    local_to_local_dofs[i].resize(dofmaps[i]->num_entity_dofs(0));

    // Resize local dof map vector
    global_dofs[i].resize(dofmaps[i]->num_entity_dofs(0));

    // Get size of local dofs
    local_dof_size[i] = dofmaps[i]->ownership_range().second
      - dofmaps[i]->ownership_range().first;

    // Get pointer to global dofs
    global_dofs_p[i].set(global_dofs[i]);
  }

  // Vector to hold dof map for a cell
  std::vector<ArrayView<const dolfin::la_index>> dofs(form_rank);

  // Exterior point integral
  const ufc::vertex_integral* integral
    = ufc.default_vertex_integral.get();

  // Check whether integral is domain-dependent
  bool use_domains = domains && !domains->empty();

  // MPI rank
  const unsigned int my_mpi_rank = MPI::rank(mesh.mpi_comm());

  // Assemble over vertices
  ufc::cell ufc_cell;
  std::vector<double> coordinate_dofs;
  Progress p(AssemblerBase::progress_message(A.rank(), "vertices"),
             mesh.num_vertices());
  for (VertexIterator vert(mesh); !vert.end(); ++vert)
  {
    // Get integral for sub domain (if any)
    if (use_domains)
      integral = ufc.get_vertex_integral((*domains)[*vert]);

    // Skip integral if zero
    if (!integral)
      continue;

    // Check if assembling a scalar and a vertex is shared
    if (form_rank == 0 && has_shared_vertices)
    {
      // Find shared processes for this global vertex
      std::map<std::int32_t, std::set<unsigned int>>::const_iterator e;
      e = shared_vertices.find(vert->index());

      // If vertex is shared and this rank is not the lowest do not
      // include the contribution from this vertex to scalar sum
      if (e != shared_vertices.end())
      {
        bool skip_vertex = false;
        std::set<unsigned int>::const_iterator it;
        for (it = e->second.begin(); it != e->second.end(); it++)
        {
          // Check if a shared vertex has a lower process rank
          if (*it < my_mpi_rank)
          {
            skip_vertex = true;
            break;
          }
        }

        if (skip_vertex)
          continue;
      }
    }

    // Get mesh cell to which mesh vertex belongs (pick first)
    Cell mesh_cell(mesh, vert->entities(D)[0]);

    // Check that cell is not a ghost
    dolfin_assert(!mesh_cell.is_ghost());

    // Get local index of vertex with respect to the cell
    const std::size_t local_vertex = mesh_cell.index(*vert);

    // Update UFC cell
    mesh_cell.get_cell_data(ufc_cell);
    mesh_cell.get_coordinate_dofs(coordinate_dofs);

    // Update UFC object
    ufc.update(mesh_cell, coordinate_dofs, ufc_cell,
               integral->enabled_coefficients());

    // Tabulate vertex tensor
    integral->tabulate_tensor(ufc.A.data(),
                              ufc.w(),
                              coordinate_dofs.data(),
                              local_vertex,
                              ufc_cell.orientation);

    // For rank 1 and 2 tensors we need to check if tabulated dofs for
    // the test space is within the local range
    bool owns_all_dofs = true;
    for (std::size_t i = 0; i < form_rank; ++i)
    {
      // Get local-to-global dof maps for cell
      auto dmap = dofmaps[i]->cell_dofs(mesh_cell.index());
      dofs[i].set(dmap.size(), dmap.data());

      // Get local dofs of the local vertex
      dofmaps[i]->tabulate_entity_dofs(local_to_local_dofs[i], 0, local_vertex);

      // Copy cell dofs to local dofs and check owner ship range
      for (std::size_t j = 0; j < local_to_local_dofs[i].size(); ++j)
      {
        global_dofs[i][j] = dofs[i][local_to_local_dofs[i][j]];

        // It is the dofs for the test space that determines if a dof
        // is owned by a process, therefore i==0
        if (i == 0 && global_dofs[i][j] >= local_dof_size[i])
        {
          owns_all_dofs = false;
          break;
        }
      }
    }

    // If not owning all dofs
    if (!owns_all_dofs)
      continue;

    // Scalar
    if (form_rank == 0)
    {
      // Add entries to global tensor
      A.add_local(ufc.A.data(), dofs);
    }
    else if (form_rank == 1)
    {
      // Copy tabulated tensor to local value vector
      for (std::size_t i = 0; i < local_to_local_dofs[0].size(); ++i)
        local_values[i] = ufc.A[local_to_local_dofs[0][i]];

      // Add local entries to global tensor
      A.add_local(local_values.data(), global_dofs_p);
    }
    else
    {
      // Copy tabulated tensor to local value vector
      const std::size_t num_cols = dofs[1].size();
      for (std::size_t i = 0; i < local_to_local_dofs[0].size(); ++i)
      {
        for (std::size_t j = 0; j < local_to_local_dofs[1].size(); ++j)
        {
          local_values[i*local_to_local_dofs[1].size() + j]
            = ufc.A[local_to_local_dofs[0][i]*num_cols
                    + local_to_local_dofs[1][j]];
        }
      }

      // Add local entries to global tensor
      A.add_local(local_values.data(), global_dofs_p);
    }

    p++;
  }
}
//-----------------------------------------------------------------------------
