/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2008 - 2022 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Daniel Arndt, Matthias Maier, 2015
 *
 * Based on step-22 by Wolfgang Bangerth and Martin Kronbichler
 */

// This example program is a slight modification of step-22 running in parallel
// using Trilinos to demonstrate the usage of periodic boundary conditions in
// deal.II. We thus omit to discuss the majority of the source code and only
// comment on the parts that deal with periodicity constraints. For the rest
// have a look at step-22 and the full source code at the bottom.

// In order to implement periodic boundary conditions only two functions
// have to be modified:
// - <code>StokesProblem<dim>::setup_dofs()</code>:
//   To populate an AffineConstraints object with periodicity constraints
// - <code>StokesProblem<dim>::create_mesh()</code>:
//   To supply a distributed triangulation with periodicity information.
//
// The rest of the program is identical to step-22, so let us skip this part
// and only show these two functions in the following. (The full program can be
// found in the "Plain program" section below, though.)


// @cond SKIP
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/base/geometry_info.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/grid/cell_id.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/block_sparsity_pattern.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <mpi.h>

namespace Step45
{
  using namespace dealii;

  template <int dim>
  class StokesProblem
  {
  public:
    StokesProblem(const unsigned int degree);
    void run();

  private:
    void create_mesh();
    bool resolve_improved();
    bool resolve_improved_2();
    void setup_dofs();
    void assemble_system();
    void solve();
    void output_results(const unsigned int refinement_cycle) const;
    void refine_mesh();
    void resolve_hanging_on_periodic_boundary();
    void resolve_refine_flags_on_periodic_boundary();
    void exchange_refinement_flags();
    int count_periodic_faces();


    const unsigned int degree;

    MPI_Comm mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;
    FESystem<dim>                             fe;
    DoFHandler<dim>                           dof_handler;

    AffineConstraints<double> constraints;
    std::vector<IndexSet>     owned_partitioning;
    std::vector<IndexSet>     relevant_partitioning;

    TrilinosWrappers::BlockSparseMatrix system_matrix;

    TrilinosWrappers::BlockSparseMatrix preconditioner_matrix;

    TrilinosWrappers::MPI::BlockVector solution;
    TrilinosWrappers::MPI::BlockVector system_rhs;

    ConditionalOStream pcout;

    MappingQ<dim> mapping;
  };



  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues()
      : Function<dim>(dim + 1)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  value) const override;
  };

  
  template <int dim>
  double BoundaryValues<dim>::value(const Point<dim> & /*p*/,
                                    const unsigned int component) const
  {
    (void)component;
    Assert(component < this->n_components,
           ExcIndexRange(component, 0, this->n_components));

    return 0;
  }


  template <int dim>
  void BoundaryValues<dim>::vector_value(const Point<dim> &p,
                                         Vector<double> &  values) const
  {
    for (unsigned int c = 0; c < this->n_components; ++c)
      values(c) = BoundaryValues<dim>::value(p, c);
  }


  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide()
      : Function<dim>(dim + 1)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  value) const override;
  };


  template <int dim>
  double RightHandSide<dim>::value(const Point<dim> & p,
                                   const unsigned int component) const
  {
    const Point<dim> center(0.75, 0.1);
    const double     r = (p - center).norm();

    if (component == 0)
      return std::exp(-100. * r * r);
    return 0;
  }


  template <int dim>
  void RightHandSide<dim>::vector_value(const Point<dim> &p,
                                        Vector<double> &  values) const
  {
    for (unsigned int c = 0; c < this->n_components; ++c)
      values(c) = RightHandSide<dim>::value(p, c);
  }



  template <class MatrixType, class PreconditionerType>
  class InverseMatrix : public Subscriptor
  {
  public:
    InverseMatrix(const MatrixType &        m,
                  const PreconditionerType &preconditioner,
                  const IndexSet &          locally_owned,
                  const MPI_Comm            mpi_communicator);

    void vmult(TrilinosWrappers::MPI::Vector &      dst,
               const TrilinosWrappers::MPI::Vector &src) const;

  private:
    const SmartPointer<const MatrixType>         matrix;
    const SmartPointer<const PreconditionerType> preconditioner;

    const MPI_Comm *                      mpi_communicator;
    mutable TrilinosWrappers::MPI::Vector tmp;
  };



  template <class MatrixType, class PreconditionerType>
  InverseMatrix<MatrixType, PreconditionerType>::InverseMatrix(
    const MatrixType &        m,
    const PreconditionerType &preconditioner,
    const IndexSet &          locally_owned,
    const MPI_Comm            mpi_communicator)
    : matrix(&m)
    , preconditioner(&preconditioner)
    , mpi_communicator(&mpi_communicator)
    , tmp(locally_owned, mpi_communicator)
  {}



  template <class MatrixType, class PreconditionerType>
  void InverseMatrix<MatrixType, PreconditionerType>::vmult(
    TrilinosWrappers::MPI::Vector &      dst,
    const TrilinosWrappers::MPI::Vector &src) const
  {
    SolverControl              solver_control(src.size(), 1e-6 * src.l2_norm());
    TrilinosWrappers::SolverCG cg(solver_control,
                                  TrilinosWrappers::SolverCG::AdditionalData());

    tmp = 0.;
    cg.solve(*matrix, tmp, src, *preconditioner);
    dst = tmp;
  }



  template <class PreconditionerType>
  class SchurComplement : public TrilinosWrappers::SparseMatrix
  {
  public:
    SchurComplement(const TrilinosWrappers::BlockSparseMatrix &system_matrix,
                    const InverseMatrix<TrilinosWrappers::SparseMatrix,
                                        PreconditionerType> &  A_inverse,
                    const IndexSet &                           owned_pres,
                    const MPI_Comm mpi_communicator);

    void vmult(TrilinosWrappers::MPI::Vector &      dst,
               const TrilinosWrappers::MPI::Vector &src) const;

  private:
    const SmartPointer<const TrilinosWrappers::BlockSparseMatrix> system_matrix;
    const SmartPointer<
      const InverseMatrix<TrilinosWrappers::SparseMatrix, PreconditionerType>>
                                          A_inverse;
    mutable TrilinosWrappers::MPI::Vector tmp1, tmp2;
  };



  template <class PreconditionerType>
  SchurComplement<PreconditionerType>::SchurComplement(
    const TrilinosWrappers::BlockSparseMatrix &system_matrix,
    const InverseMatrix<TrilinosWrappers::SparseMatrix, PreconditionerType>
      &             A_inverse,
    const IndexSet &owned_vel,
    const MPI_Comm  mpi_communicator)
    : system_matrix(&system_matrix)
    , A_inverse(&A_inverse)
    , tmp1(owned_vel, mpi_communicator)
    , tmp2(tmp1)
  {}



  template <class PreconditionerType>
  void SchurComplement<PreconditionerType>::vmult(
    TrilinosWrappers::MPI::Vector &      dst,
    const TrilinosWrappers::MPI::Vector &src) const
  {
    system_matrix->block(0, 1).vmult(tmp1, src);
    A_inverse->vmult(tmp2, tmp1);
    system_matrix->block(1, 0).vmult(dst, tmp2);
  }



  template <int dim>
  StokesProblem<dim>::StokesProblem(const unsigned int degree)
    : degree(degree)
    , mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator)
    , fe(FE_Q<dim>(degree + 1), dim, FE_Q<dim>(degree), 1)
    , dof_handler(triangulation)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    , mapping(degree + 1)
  {}
  // @endcond
  //
  // @sect3{Setting up periodicity constraints on distributed triangulations}
  template <int dim>
  void StokesProblem<dim>::create_mesh()
  {
    Point<dim>   center;
    const double inner_radius = .5;
    const double outer_radius = 1.;

    // GridGenerator::quarter_hyper_shell(
    //   triangulation, center, inner_radius, outer_radius, 0, true);
    GridGenerator::hyper_cube(triangulation,0,1,true);
    
    


    // Before we can prescribe periodicity constraints, we need to ensure that
    // cells on opposite sides of the domain but connected by periodic faces are
    // part of the ghost layer if one of them is stored on the local processor.
    // At this point we need to think about how we want to prescribe
    // periodicity. The vertices $\text{vertices}_2$ of a face on the left
    // boundary should be matched to the vertices $\text{vertices}_1$ of a face
    // on the lower boundary given by $\text{vertices}_2=R\cdot
    // \text{vertices}_1+b$ where the rotation matrix $R$ and the offset $b$ are
    // given by
    // @f{align*}
    // R=\begin{pmatrix}
    // 0&1\\-1&0
    // \end{pmatrix},
    // \quad
    // b=\begin{pmatrix}0&0\end{pmatrix}.
    // @f}
    // The data structure we are saving the resulting information into is here
    // based on the Triangulation.
    std::vector<GridTools::PeriodicFacePair<
      typename parallel::distributed::Triangulation<dim>::cell_iterator>>
      periodicity_vector;

    FullMatrix<double> rotation_matrix(dim);
    rotation_matrix[0][1] = 1.;
    rotation_matrix[1][0] = -1.;
    
    GridTools::collect_periodic_faces(triangulation,
                                      2,
                                      3,
                                      1,
                                      periodicity_vector
                                      );

    // Now telling the triangulation about the desired periodicity is
    // particularly easy by just calling
    // parallel::distributed::Triangulation::add_periodicity.
    triangulation.add_periodicity(periodicity_vector);

    triangulation.refine_global(2);
    output_results(0);
    const auto refinement_subdomain_predicate = [&](const auto &cell) {
      return (cell->center()(1) > 0.5 && cell->center()(1)<.75 && cell->center()(2)<.25);
    };

    for (auto &cell :
         triangulation.active_cell_iterators() | refinement_subdomain_predicate){
        cell->set_refine_flag();
       
      }

    //  triangulation.add_periodicity(periodicity_vector);



    // resolve_hanging_on_periodic_boundary();
    triangulation.execute_coarsening_and_refinement();
    

    output_results(1);

  }


  template <int dim>
  void StokesProblem<dim>::setup_dofs()
  {
    dof_handler.distribute_dofs(fe);

    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    DoFRenumbering::component_wise(dof_handler, block_component);

    const std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    const unsigned int n_u = dofs_per_block[0], n_p = dofs_per_block[1];

    {
      owned_partitioning.clear();
      IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
      owned_partitioning.push_back(locally_owned_dofs.get_view(0, n_u));
      owned_partitioning.push_back(locally_owned_dofs.get_view(n_u, n_u + n_p));

      relevant_partitioning.clear();
      const IndexSet locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(dof_handler);
      relevant_partitioning.push_back(locally_relevant_dofs.get_view(0, n_u));
      relevant_partitioning.push_back(
        locally_relevant_dofs.get_view(n_u, n_u + n_p));

      constraints.clear();
      constraints.reinit(locally_relevant_dofs);

      const FEValuesExtractors::Vector velocities(0);

      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      VectorTools::interpolate_boundary_values(mapping,
                                               dof_handler,
                                               0,
                                               BoundaryValues<dim>(),
                                               constraints,
                                               fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(mapping,
                                               dof_handler,
                                               1,
                                               BoundaryValues<dim>(),
                                               constraints,
                                               fe.component_mask(velocities));

      // After we provided the mesh with the necessary information for the
      // periodicity constraints, we are now able to actual create them. For
      // describing the matching we are using the same approach as before, i.e.,
      // the $\text{vertices}_2$ of a face on the left boundary should be
      // matched to the vertices
      // $\text{vertices}_1$ of a face on the lower boundary given by
      // $\text{vertices}_2=R\cdot \text{vertices}_1+b$ where the rotation
      // matrix $R$ and the offset $b$ are given by
      // @f{align*}
      // R=\begin{pmatrix}
      // 0&1\\-1&0
      // \end{pmatrix},
      // \quad
      // b=\begin{pmatrix}0&0\end{pmatrix}.
      // @f}
      // These two objects not only describe how faces should be matched but
      // also in which sense the solution should be transformed from
      // $\text{face}_2$ to
      // $\text{face}_1$.
      FullMatrix<double> rotation_matrix(dim);
      rotation_matrix[0][1] = 1.;
      rotation_matrix[1][0] = -1.;

      Tensor<1, dim> offset;

      // For setting up the constraints, we first store the periodicity
      // information in an auxiliary object of type
      // <code>std::vector@<GridTools::PeriodicFacePair<typename
      // DoFHandler@<dim@>::%cell_iterator@> </code>. The periodic boundaries
      // have the boundary indicators 2 (x=0) and 3 (y=0). All the other
      // parameters we have set up before. In this case the direction does not
      // matter. Due to $\text{vertices}_2=R\cdot \text{vertices}_1+b$ this is
      // exactly what we want.
      std::vector<
        GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator>>
        periodicity_vector;

      const unsigned int direction = 1;

      std::cout<<"here";

      GridTools::collect_periodic_faces(dof_handler,
                                        2,
                                        3,
                                        direction,
                                        periodicity_vector,
                                        offset,
                                        rotation_matrix);

      // Next, we need to provide information on which vector valued components
      // of the solution should be rotated. Since we choose here to just
      // constraint the velocity and this starts at the first component of the
      // solution vector, we simply insert a 0:
      std::vector<unsigned int> first_vector_components;
      first_vector_components.push_back(0);

      // After setting up all the information in periodicity_vector all we have
      // to do is to tell make_periodicity_constraints to create the desired
      // constraints.
      DoFTools::make_periodicity_constraints<dim, dim>(periodicity_vector,
                                                       constraints,
                                                       fe.component_mask(
                                                         velocities),
                                                       first_vector_components);
    }

    constraints.close();

    {
      TrilinosWrappers::BlockSparsityPattern bsp(owned_partitioning,
                                                 owned_partitioning,
                                                 relevant_partitioning,
                                                 mpi_communicator);

      Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
      for (unsigned int c = 0; c < dim + 1; ++c)
        for (unsigned int d = 0; d < dim + 1; ++d)
          if (!((c == dim) && (d == dim)))
            coupling[c][d] = DoFTools::always;
          else
            coupling[c][d] = DoFTools::none;

      DoFTools::make_sparsity_pattern(dof_handler,
                                      coupling,
                                      bsp,
                                      constraints,
                                      false,
                                      Utilities::MPI::this_mpi_process(
                                        mpi_communicator));

      bsp.compress();

      system_matrix.reinit(bsp);
    }

    {
      TrilinosWrappers::BlockSparsityPattern preconditioner_bsp(
        owned_partitioning,
        owned_partitioning,
        relevant_partitioning,
        mpi_communicator);

      Table<2, DoFTools::Coupling> preconditioner_coupling(dim + 1, dim + 1);
      for (unsigned int c = 0; c < dim + 1; ++c)
        for (unsigned int d = 0; d < dim + 1; ++d)
          if ((c == dim) && (d == dim))
            preconditioner_coupling[c][d] = DoFTools::always;
          else
            preconditioner_coupling[c][d] = DoFTools::none;

      DoFTools::make_sparsity_pattern(dof_handler,
                                      preconditioner_coupling,
                                      preconditioner_bsp,
                                      constraints,
                                      false,
                                      Utilities::MPI::this_mpi_process(
                                        mpi_communicator));

      preconditioner_bsp.compress();

      preconditioner_matrix.reinit(preconditioner_bsp);
    }

    system_rhs.reinit(owned_partitioning, mpi_communicator);
    solution.reinit(owned_partitioning,
                    relevant_partitioning,
                    mpi_communicator);
  }

  // The rest of the program is then again identical to step-22. We will omit
  // it here now, but as before, you can find these parts in the "Plain program"
  // section below.

  // @cond SKIP
  template <int dim>
  void StokesProblem<dim>::assemble_system()
  {
    system_matrix         = 0.;
    system_rhs            = 0.;
    preconditioner_matrix = 0.;

    QGauss<dim> quadrature_formula(degree + 2);

    FEValues<dim> fe_values(mapping,
                            fe,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_preconditioner_matrix(dofs_per_cell,
                                                   dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const RightHandSide<dim>    right_hand_side;
    std::vector<Vector<double>> rhs_values(n_q_points, Vector<double>(dim + 1));

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    std::vector<SymmetricTensor<2, dim>> symgrad_phi_u(dofs_per_cell);
    std::vector<double>                  div_phi_u(dofs_per_cell);
    std::vector<double>                  phi_p(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          local_matrix                = 0;
          local_preconditioner_matrix = 0;
          local_rhs                   = 0;

          right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                            rhs_values);

          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                  symgrad_phi_u[k] =
                    fe_values[velocities].symmetric_gradient(k, q);
                  div_phi_u[k] = fe_values[velocities].divergence(k, q);
                  phi_p[k]     = fe_values[pressure].value(k, q);
                }

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j <= i; ++j)
                    {
                      local_matrix(i, j) +=
                        (symgrad_phi_u[i] * symgrad_phi_u[j] // diffusion
                         - div_phi_u[i] * phi_p[j]           // pressure force
                         - phi_p[i] * div_phi_u[j])          // divergence
                        * fe_values.JxW(q);

                      local_preconditioner_matrix(i, j) +=
                        (phi_p[i] * phi_p[j]) * fe_values.JxW(q);
                    }

                  const unsigned int component_i =
                    fe.system_to_component_index(i).first;
                  local_rhs(i) += fe_values.shape_value(i, q)  //
                                  * rhs_values[q](component_i) //
                                  * fe_values.JxW(q);
                }
            }

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
              {
                local_matrix(i, j) = local_matrix(j, i);
                local_preconditioner_matrix(i, j) =
                  local_preconditioner_matrix(j, i);
              }

          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(local_matrix,
                                                 local_rhs,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_rhs);
          constraints.distribute_local_to_global(local_preconditioner_matrix,
                                                 local_dof_indices,
                                                 preconditioner_matrix);
        }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    pcout << "   Computing preconditioner..." << std::endl << std::flush;
  }



  template <int dim>
  void StokesProblem<dim>::solve()
  {
    TrilinosWrappers::PreconditionJacobi A_preconditioner;
    A_preconditioner.initialize(system_matrix.block(0, 0));

    const InverseMatrix<TrilinosWrappers::SparseMatrix,
                        TrilinosWrappers::PreconditionJacobi>
      A_inverse(system_matrix.block(0, 0),
                A_preconditioner,
                owned_partitioning[0],
                mpi_communicator);

    TrilinosWrappers::MPI::BlockVector tmp(owned_partitioning,
                                           mpi_communicator);

    {
      TrilinosWrappers::MPI::Vector schur_rhs(owned_partitioning[1],
                                              mpi_communicator);
      A_inverse.vmult(tmp.block(0), system_rhs.block(0));
      system_matrix.block(1, 0).vmult(schur_rhs, tmp.block(0));
      schur_rhs -= system_rhs.block(1);

      SchurComplement<TrilinosWrappers::PreconditionJacobi> schur_complement(
        system_matrix, A_inverse, owned_partitioning[0], mpi_communicator);

      SolverControl solver_control(solution.block(1).size(),
                                   1e-6 * schur_rhs.l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector> cg(solver_control);

      TrilinosWrappers::PreconditionAMG preconditioner;
      preconditioner.initialize(preconditioner_matrix.block(1, 1));

      InverseMatrix<TrilinosWrappers::SparseMatrix,
                    TrilinosWrappers::PreconditionAMG>
        m_inverse(preconditioner_matrix.block(1, 1),
                  preconditioner,
                  owned_partitioning[1],
                  mpi_communicator);

      cg.solve(schur_complement, tmp.block(1), schur_rhs, preconditioner);

      constraints.distribute(tmp);
      solution.block(1) = tmp.block(1);
    }

    {
      system_matrix.block(0, 1).vmult(tmp.block(0), tmp.block(1));
      tmp.block(0) *= -1;
      tmp.block(0) += system_rhs.block(0);

      A_inverse.vmult(tmp.block(0), tmp.block(0));

      constraints.distribute(tmp);
      solution.block(0) = tmp.block(0);
    }
  }



  template <int dim>
  void
  StokesProblem<dim>::output_results(const unsigned int refinement_cycle) const
  {
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    // data_out.add_data_vector(solution,
    //                          solution_names,
    //                          DataOut<dim>::type_dof_data,
    //                          data_component_interpretation);
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record(
      "./", "sol", refinement_cycle, MPI_COMM_WORLD, 2);
  }

  

  template<int dim>
  void StokesProblem<dim>::resolve_hanging_on_periodic_boundary(){
    // output_results(1);
    std::vector<CellId> list;
    std::vector<CellId> local_boundary_cells;
    int local_count=0;
    // int* local_count=(int*)malloc (sizeof(int));
    // local_count[0]=0;
    int* recvcounts;
    for(auto& cell: triangulation.active_cell_iterators()){
      if(cell->is_locally_owned()){
        cell->clear_refine_flag();

      }
    }
    
    for(auto& cell: triangulation.active_cell_iterators()){
      if(cell->is_locally_owned() && cell->at_boundary()){
        local_boundary_cells.push_back(cell->id());
        for(unsigned int i=0;i<cell->n_faces();++i){
          if(cell-> has_periodic_neighbor(i)){
            if(cell-> periodic_neighbor_is_coarser(i)){
              list.push_back(cell->periodic_neighbor(i)->id());
              ++local_count;
            }
          }
        }
      }
    }
    std::cout<<"local count is "<<local_count<<"\n";
    MPI_Barrier(MPI_COMM_WORLD);

    int* convert_list=(int*)malloc(sizeof(unsigned int)*list.size()*4);
    for(unsigned int i=0;i<list.size();++i){
      std::array<unsigned int,4> temp=list[i].to_binary<dim>();
      // std::cout<<MPI_Comm_rank(MPI_COMM_WORLD,&comm_rank)<<":";
      for(unsigned int j=0;j<4;++j){
        convert_list[4*i+j]= temp[j];
        std::cout<<convert_list[4*i+j]<<" ";
      }
      std::cout<<"\n";

    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    int ranks; //number of ranks

    MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&comm_rank);

    if(comm_rank==0)
      std::cout<<"number of ranks"<<ranks<<"\n";

    
    int count=count_periodic_faces();
    int global_count;
    int my_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&my_id);
    MPI_Reduce(&count,&global_count,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Bcast(&global_count,1,MPI_INT,0,MPI_COMM_WORLD);
    int stride=2; //create formula for value
    std::cout<<"stride is "<<stride;
    int *displs;
    displs=(int* )malloc(ranks*sizeof(int));
    for(int i=0;i<ranks;++i){
      displs[i]=i*stride;

    }
    MPI_Barrier(MPI_COMM_WORLD);
  
    int* global_list=(int* )malloc(global_count*sizeof(int)*8);

    recvcounts=(int*)malloc(ranks*sizeof(int)); //will be used to gather cell ids
    local_count=local_count*4;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allgather(&local_count,1,MPI_INT,recvcounts,1,MPI_INT,MPI_COMM_WORLD);
    displs[0]=0;
    for(int i=1;i<ranks;++i){
      displs[i]=displs[i-1]+recvcounts[i-1];
    }
    std::cout<<"numbmer of periodic faces is "<<global_count<<"\n";
    if(comm_rank==0){
      for(int i=0;i<ranks;++i){
        std::cout<<"displs["<<i<<"]="<<displs[i]<<"\n";
      }
    }
    std::cout<<"\n list size is "<<list.size();
    MPI_Barrier(MPI_COMM_WORLD);
 

    std::cout<<"convert_list: \n";

    for(int i=0;i<list.size()*4;++i){
      std::cout<<convert_list[i]<<" ";
    }
    MPI_Allgatherv(convert_list,list.size()*4,MPI_INT,global_list,recvcounts,displs,MPI_INT,MPI_COMM_WORLD); 
  
      int sum_recvcounts=0;               //fix indentation
      for(int i=0;i<ranks;++i){
        sum_recvcounts+=recvcounts[i];

      }
      std::cout<<"\n sum_recvcounts is "<<sum_recvcounts;
      std::cout<<"CELL IDS BELOW  \n";
      for(int i=0;i<sum_recvcounts;++i){
        std::cout<<global_list[i]<<" ";
        
      }
    
    std::vector<CellId> ids;
    for(int i=0;i<sum_recvcounts/4;++i){
      std::array<unsigned int,4> temp;
      for(int j=0;j<4;++j){
        temp[j]=global_list[4*i+j];
      }
      CellId tempid=CellId(temp);
      ids.push_back(tempid);
    }
    for(int i=0;i<ids.size();++i){
      for(int j=0;j<local_boundary_cells.size();++j){
        if (ids[i]==local_boundary_cells[j]){
          triangulation.create_cell_iterator(ids[i])->set_refine_flag();
        }
      }
    }
    
    
   
   
      
  }

  template<int dim>
  bool StokesProblem<dim>::resolve_improved(){
    bool flags_set=false;
    exchange_refinement_flags();
    for(auto &cell:triangulation.active_cell_iterators()){
      if(cell->at_boundary() && cell->refine_flag_set()){
        for(unsigned int i=0;i<cell->n_faces();++i){
          if(cell->has_periodic_neighbor(i)){
            TriaIterator<CellAccessor<dim,dim>> temp=cell->periodic_neighbor(i);
            if(temp->is_active()&&!temp->refine_flag_set()){
              temp->set_refine_flag();
              flags_set=true;
            }
          }
         
        }
      }
    }
    exchange_refinement_flags();
    return flags_set;
  }
  template<int dim>
  bool StokesProblem<dim>::resolve_improved_2(){
    bool flags_set=false;
    for(auto &cell:triangulation.active_cell_iterators()){
      if(cell->at_boundary()){
        for(int i=0;i<cell->n_faces();++i){
          if(cell->has_periodic_neighbor(i) && cell->periodic_neighbor(i)->refine_flag_set()){
            cell->set_refine_flag();
            flags_set=true;
          }
        }
      }
    }
    return flags_set;
  }

  template<int dim>
  void StokesProblem<dim>::resolve_refine_flags_on_periodic_boundary(){
      int local_count=0;
      std::vector<CellId> list;
      std::vector<CellId> local_boundary_cells;
      for(auto& cell: triangulation.active_cell_iterators()){
        if(cell->is_locally_owned() && cell->at_boundary()){
          local_boundary_cells.push_back(cell->id());
        }
      }
      for (auto &cell : triangulation.active_cell_iterators()) {
        
       if(cell->is_locally_owned()  && cell->refine_flag_set()){
         for(unsigned int i=0;i<cell->n_faces();++i){
           if(cell->has_periodic_neighbor(i)){
              
             TriaIterator< CellAccessor< dim,dim > >  neighbor=cell->periodic_neighbor(i);
             if(neighbor->is_active()){
              list.push_back(neighbor->id());
              ++local_count;
             }
           }
         }
       }
       
     }

      int* convert_list=(int*)malloc(sizeof(unsigned int)*list.size()*4);
      for(unsigned int i=0;i<list.size();++i){
      std::array<unsigned int,4> temp=list[i].to_binary<dim>();
      for(unsigned int j=0;j<4;++j){
        convert_list[4*i+j]= temp[j];
        std::cout<<convert_list[4*i+j]<<" ";
      }
      std::cout<<"\n";

    }

     
      int ranks; //number of ranks

    MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&comm_rank);

    if(comm_rank==0)
      std::cout<<"number of ranks"<<ranks<<"\n";

    
    int count=count_periodic_faces();
    int global_count;
    int my_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&my_id);
    MPI_Reduce(&count,&global_count,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Bcast(&global_count,1,MPI_INT,0,MPI_COMM_WORLD);
    int stride=2; //create formula for value
    std::cout<<"stride is "<<stride;
    int *displs;
    displs=(int* )malloc(ranks*sizeof(int));
    for(int i=0;i<ranks;++i){
      displs[i]=i*stride;

    }
    MPI_Barrier(MPI_COMM_WORLD);
  
    int* global_list=(int* )malloc(global_count*sizeof(int)*8);

    int* recvcounts=(int*)malloc(ranks*sizeof(int)); //will be used to gather cell ids
    local_count=local_count*4;
    MPI_Barrier(MPI_COMM_WORLD);
   MPI_Allgather(&local_count,1,MPI_INT,recvcounts,1,MPI_INT,MPI_COMM_WORLD);

    MPI_Allgather(&local_count,1,MPI_INT,recvcounts,1,MPI_INT,MPI_COMM_WORLD);
    displs[0]=0;
    for(int i=1;i<ranks;++i){
      displs[i]=displs[i-1]+recvcounts[i-1];
    }
    std::cout<<"numbmer of periodic faces is "<<global_count<<"\n";
    if(comm_rank==0){
      for(int i=0;i<ranks;++i){
        std::cout<<"displs["<<i<<"]="<<displs[i]<<"\n";
      }
    }
    std::cout<<"\n list size is "<<list.size();
    MPI_Barrier(MPI_COMM_WORLD);
 

    std::cout<<"convert_list: \n";

    for(int i=0;i<list.size()*4;++i){
      std::cout<<convert_list[i]<<" ";
    }
    MPI_Allgatherv(convert_list,list.size()*4,MPI_INT,global_list,recvcounts,displs,MPI_INT,MPI_COMM_WORLD); 
  
      int sum_recvcounts=0;               //fix indentation
      for(int i=0;i<ranks;++i){
        sum_recvcounts+=recvcounts[i];

      }
      std::cout<<"\n sum_recvcounts is "<<sum_recvcounts;
      std::cout<<"CELL IDS BELOW  \n";
      for(int i=0;i<sum_recvcounts;++i){
        std::cout<<global_list[i]<<" ";
        
      }
    
    std::vector<CellId> ids;
    for(int i=0;i<sum_recvcounts/4;++i){
      std::array<unsigned int,4> temp;
      for(int j=0;j<4;++j){
        temp[j]=global_list[4*i+j];
      }
      CellId tempid=CellId(temp);
      ids.push_back(tempid);
    }
    for(int i=0;i<ids.size();++i){
      for(int j=0;j<local_boundary_cells.size();++j){
        if (ids[i]==local_boundary_cells[j]){
          triangulation.create_cell_iterator(ids[i])->set_refine_flag();
        }
      }
    }

  }
  template<int dim>
  int StokesProblem<dim>::count_periodic_faces(){
    unsigned int count=0;
    for(auto &cell: triangulation.active_cell_iterators()){
      if(cell -> is_locally_owned()){
        bool neighbor_exists=false;
        for(int i=0;i<cell->n_faces();++i){
          if(cell->has_periodic_neighbor(i)){
            neighbor_exists=true;
          }
        }
        if(neighbor_exists){
          ++count;
        }
      }
    }

    return count;
  }

  //refine twice any x

  // then y between .5 and .75 

  //center z<.25

  //second predicate x>.25

  template <int dim>
  void StokesProblem<dim>::exchange_refinement_flags ()
  {
    // Communicate refinement flags on ghost cells from the owner of the
    // cell. This is necessary to get consistent refinement, as mesh
    // smoothing would undo some of the requested coarsening/refinement.

    auto pack
    = [] (const typename DoFHandler<dim>::active_cell_iterator &cell) -> std::uint8_t
    {
      if (cell->refine_flag_set())
        return 1;
      if (cell->coarsen_flag_set())
        return 2;
      return 0;
    };
    auto unpack
    = [] (const typename DoFHandler<dim>::active_cell_iterator &cell, const std::uint8_t &flag) -> void
    {
      cell->clear_coarsen_flag();
      cell->clear_refine_flag();
      if (flag==1)
        cell->set_refine_flag();
      else if (flag==2)
        cell->set_coarsen_flag();
    };

    GridTools::exchange_cell_data_to_ghosts<std::uint8_t, DoFHandler<dim>>
    (dof_handler, pack, unpack);
  }

int count=0;

  template <int dim>
  void StokesProblem<dim>::refine_mesh()
  {

    const auto refinement_subdomain_predicate = [&](const auto &cell) {
      return (cell->center()(0) < 0.25 && cell->center()(1) > 0.5 && cell->center()(1)< .75 && cell->center()(2)<.25);
    };

   for (auto &cell :
       triangulation.active_cell_iterators() | refinement_subdomain_predicate){
          cell->set_refine_flag();
       }
    bool changed=false;
    do{
      triangulation.prepare_coarsening_and_refinement();
      exchange_refinement_flags();
      changed=resolve_improved();
      ++count;
      
    }while(changed);
    std::cout<<count;
    triangulation.execute_coarsening_and_refinement();
  }

// while (changed)
// {
//   prepare();
//   exchange();
//   for all of my cells, if a periodic neighbor is marked refinement, set_refine_flag() and changed=true;

// };

  //  resolve_refine_flags_on_periodic_boundary();

  
  

  template <int dim>
  void StokesProblem<dim>::run()
  {
    create_mesh();
    // std::cout<<count_periodic_faces();

    for (unsigned int refinement_cycle = 2; refinement_cycle < 5;
         ++refinement_cycle)
      {
        pcout << "Refinement cycle " << refinement_cycle << std::endl;

        refine_mesh();

        // setup_dofs();

        output_results(refinement_cycle);

        pcout << std::endl;
      }
  }
} // namespace Step45


int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace Step45;
      
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 8);
      StokesProblem<3>                 flow_problem(1);
      

    
    //   MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    //   MPI_Comm_size(MPI_COMM_WORLD,&size);
    //   printf("process %d of %d\n",rank,size);
      flow_problem.run();
      return 0;
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
// @endcond
