package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
)

/*
Perceptron struct defining all of the info
that pertains to a perceptron
*/
type Perceptron struct {
	weights *mat.VecDense
	bias    float64
}

// initialize Perceptron
func newPerceptron() *Perceptron {
	p := Perceptron{
		weights: mat.NewVecDense(2, nil),
		bias:    0,
	}
	return &p
}

// format and print matrices
func printMat(M mat.Matrix) {
	f := mat.Formatted(M, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", f)
}

// train Perceptron
func (p *Perceptron) train(x *mat.Dense, y *mat.VecDense) {
	// append bias to weight vector for computation
	d := p.weights.Len()
	c := mat.NewVecDense(d+1, nil)
	c.CopyVec(p.weights)
	c.SetVec(d, p.bias)
	fmt.Println("New synth weight vec:")
	printMat(c)
	// end condition
	done := false
	fmt.Println("Starting training...")
	for done != true {
		// track number of correctly classified points
		numRight := 0
		// loop over labels
		d := y.Len()
		for i := 0; i < d; i++ {
			// get this label
			l := y.AtVec(i)
			// get this input vector
			a := x.RowView(i)
			// append 1 to accomodate combined weight/bias vec
			a_new := mat.NewVecDense(a.Len()+1, nil)
			a_new.CopyVec(a)
			a_new.SetVec(a.Len(), 1)
			// compute dot product
			//prod := mat.Dot(a, p.weights)
			prod := mat.Dot(a_new, c)
			// update
			if l*prod > 0 {
				fmt.Printf("Input %d passed\n", i)
				numRight = numRight + 1
			} else {
				fmt.Printf("Input %d failed\n", i)
				//p.weights.AddScaledVec(p.weights, l, a)
				c.AddScaledVec(c, l, a_new)
				fmt.Println("updated weight vector: ")
				printMat(c)
				break
			}
		}
		if numRight == d {
			fmt.Println("Final weight vector: ")
			printMat(c)
			p.weights.CopyVec(c)
			p.bias = c.AtVec(p.weights.Len())
			done = true
		}
	}
}

// main
func main() {
	// define inputs
	input := mat.NewDense(3, 2, []float64{
		-1.0, 1.0,
		0.0, -1.0,
		10.0, 1.0,
	})
	// define labels
	labels := mat.NewVecDense(3, []float64{1.0, -1.0, 1.0})
	// initialize Perceptron
	p := newPerceptron()

	// check matrices
	fmt.Println("Input:")
	printMat(input)
	fmt.Println("Labels:")
	printMat(labels)
	fmt.Println("Initial weights:")
	printMat(p.weights)
	fmt.Printf("Initial bias: %f\n", p.bias)
	p.train(input, labels)
	fmt.Println("Converged weights:")
	printMat(p.weights)
	fmt.Printf("Converged bias: %f\n", p.bias)
}
