// package main holds a customization of the mip incentive allocation
// template.
package main

import (
	"context"
	"log"
	"math"

	"github.com/nextmv-io/sdk/mip"
	"github.com/nextmv-io/sdk/run"
	"github.com/nextmv-io/sdk/run/schema"
)

// This template demonstrates how to solve a Mixed Integer Programming problem.
// To solve a mixed integer problem is to optimize a linear objective function
// of many variables, subject to linear constraints. We demonstrate this by
// solving a Renovation allocation problem.
func main() {
	err := run.CLI(solver).Run(context.Background())
	if err != nil {
		log.Fatal(err)
	}
}

// The options for the solver.
type options struct {
	Limits mip.Limits `json:"limits,omitempty"`
}

// Input of the problem.
type input struct {
	// Properties for the problem including name and a cost/effect pair per
	// available Renovation.
	Properties []struct {
		ID string `json:"id"`
		// Renovations that can be applied to the property.
		Renovations []struct {
			ID string `json:"id"`
			// Positive effect of the Renovation.
			Effect float64 `json:"effect"`
			// Cost of the Renovation that will be subtracted from the budget.
			Cost float64 `json:"cost"`
		} `json:"renovations"`
	} `json:"properties"`
	Budget int `json:"budget"`
}

// assignments is used to print the solutio∑n and represents the
// combination of a property with the assigned Renovation.
type assignments struct {
	Property     string  `json:"property"`
	RenovationID string  `json:"Renovation_id"`
	Cost         float64 `json:"cost"`
	Effect       float64 `json:"effect"`
}

// solution represents the decisions made by the solver.
type solution struct {
	Assignments []assignments `json:"assignments,omitempty"`
}

func solver(_ context.Context, input input, options options) (schema.Output, error) {
	// We start by creating a MIP model.
	m := mip.NewModel()

	// We want to maximize the value of the problem.
	m.Objective().SetMaximize()

	// This constraint ensures the budget of the will not be exceeded.
	budgetConstraint := m.NewConstraint(
		mip.LessThanOrEqual,
		float64(input.Budget),
	)

	// Create a map of property ID to a slice of decision variables, one for each
	// Renovation.
	propertyRenovationVariables := make(map[string][]mip.Var, len(input.Properties))
	for _, property := range input.Properties {
		// For each property, create the slice of variables based on the number of
		// Renovations.
		propertyRenovationVariables[property.ID] = make([]mip.Var, len(property.Renovations))

		// This constraint ensures that each property is assigned at most three
		// Renovations.
		countRenovationConstraint := m.NewConstraint(mip.LessThanOrEqual, 3.0)
		for i, Renovation := range property.Renovations {
			// For each Renovation, create a binary decision variable.
			propertyRenovationVariables[property.ID][i] = m.NewBool()

			// Set the term of the variable on the objective, based on the
			// effect the Renovation has on the property.
			m.Objective().NewTerm(
				Renovation.Effect,
				propertyRenovationVariables[property.ID][i],
			)

			// Set the term of the variable on the budget constraint, based on
			// the cost of the Renovation for the property.
			budgetConstraint.NewTerm(
				Renovation.Cost,
				propertyRenovationVariables[property.ID][i],
			)

			// Set the term of the variable on the constraint that controls the
			// number of Renovations per property.
			countRenovationConstraint.NewTerm(1, propertyRenovationVariables[property.ID][i])
		}
	}

	// We create a solver using the 'highs' provider.
	solver, err := mip.NewSolver("highs", m)
	if err != nil {
		return schema.Output{}, err
	}

	// We create the solve options we will use.
	solveOptions := mip.NewSolveOptions()

	// Limit the solve to a maximum duration.
	if err = solveOptions.SetMaximumDuration(options.Limits.Duration); err != nil {
		return schema.Output{}, err
	}

	// Set the relative gap to 0% (highs' default is 5%)
	if err = solveOptions.SetMIPGapRelative(0); err != nil {
		return schema.Output{}, err
	}

	// Set verbose level to see a more detailed output
	solveOptions.SetVerbosity(mip.Off)

	// Solve the model and get the solution.
	solution, err := solver.Solve(solveOptions)
	if err != nil {
		return schema.Output{}, err
	}

	// Format the solution into the desired output format and add custom
	// statistics.
	output := mip.Format(options, format(input, solution, propertyRenovationVariables), solution)
	output.Statistics.Result.Custom = mip.DefaultCustomResultStatistics(m, solution)

	return output, nil
}

// format the solution from the solver into the desired output format.
func format(
	input input,
	solverSolution mip.Solution,
	propertyRenovationVariables map[string][]mip.Var,
) solution {
	if !solverSolution.IsOptimal() && !solverSolution.IsSubOptimal() {
		return solution{}
	}

	assigned := []assignments{}
	for i, property := range input.Properties {
		for j := range property.Renovations {
			// If the variable is not assigned, skip it.
			if int(math.Round(
				solverSolution.Value(propertyRenovationVariables[property.ID][j])),
			) < 1 {
				continue
			}

			assigned = append(
				assigned,
				assignments{
					Cost:         input.Properties[i].Renovations[j].Cost,
					Effect:       input.Properties[i].Renovations[j].Effect,
					Property:     property.ID,
					RenovationID: input.Properties[i].Renovations[j].ID,
				},
			)
		}
	}

	return solution{
		Assignments: assigned,
	}
}
