package main

import (
	"fmt"
	"math/rand"
	"sort"
	"time"
)

type Layer interface {
	Copy() Layer
	GetValues() []byte
	Size() int
}

// A node that accumulates a value from one or more connections to the previous layer.
type Node struct {
	Inputs []Edge
}

// A single connection between two nodes in two adjacent layers.
type Edge struct {
	Index    int
	And, Xor byte
}

// A layer that is inferred from the previous layer ("left").
type InferredLayer struct {
	Nodes []Node
	Left  Layer
}

func (l InferredLayer) Copy() Layer {
	nodes := make([]Node, len(l.Nodes))
	for i, n := range l.Nodes {
		nodes[i] = Node{Inputs: make([]Edge, len(n.Inputs))}
		copy(nodes[i].Inputs, n.Inputs)
	}
	return &InferredLayer{
		Nodes: nodes,
		Left:  l.Left.Copy(),
	}
}

func (l InferredLayer) GetValues() []byte {
	lv := l.Left.GetValues()
	v := make([]byte, l.Size())
	for i, node := range l.Nodes {
		for _, input := range node.Inputs {
			v[i] ^= lv[input.Index]&input.And ^ input.Xor
		}
	}
	return v
}

func (l *InferredLayer) Mutate(rarity int) {
	for i := range l.Nodes {
		for j := range l.Nodes[i].Inputs {
			if rand.Intn(rarity) == 0 {
				continue
			}
			var r uint64
			r = rand.Uint64()
			l.Nodes[i].Inputs[j].And |= byte((r >> 56) & (r >> 48) & (r >> 40) & (r >> 32) & (r >> 24) & (r >> 16) & (r >> 8) & r)
			r = rand.Uint64()
			l.Nodes[i].Inputs[j].And &= byte((r >> 56) | (r >> 48) | (r >> 40) | (r >> 32) | (r >> 24) | (r >> 16) | (r >> 8) | r)
			r = rand.Uint64()
			l.Nodes[i].Inputs[j].Xor |= byte((r >> 56) & (r >> 48) & (r >> 40) & (r >> 32) & (r >> 24) & (r >> 16) & (r >> 8) & r)
			r = rand.Uint64()
			l.Nodes[i].Inputs[j].Xor &= byte((r >> 56) | (r >> 48) | (r >> 40) | (r >> 32) | (r >> 24) | (r >> 16) | (r >> 8) | r)
		}
	}
	if il, ok := l.Left.(*InferredLayer); ok {
		il.Mutate(rarity)
	}
}

func (l InferredLayer) Size() int {
	return len(l.Nodes)
}

// Utility layer for keeping score.
type ScoredLayer struct {
	*InferredLayer
	Score int
}

func (l ScoredLayer) Copy() Layer {
	return &ScoredLayer{l.InferredLayer.Copy().(*InferredLayer), 0}
}

// Non-trainable layer (i.e., input).
type StaticLayer []byte

func (l StaticLayer) Copy() Layer {
	// No-op for now.
	return l
}

func (l StaticLayer) GetValues() []byte {
	return l
}

func (l StaticLayer) Size() int {
	return len(l)
}

func main() {
	rand.Seed(time.Now().UnixNano())

	in := StaticLayer{
		0, 0, 0,
		0, 0, 0,
		0, 0, 0,
	}

	newNetwork := func() *InferredLayer {
		var l Layer = in
		for i := 0; i < 10; i++ {
			l = NewFullyConnectedLayer(l, 9)
		}
		return NewFullyConnectedLayer(l, 9)
	}

	pop := []ScoredLayer{}
	for i := 0; i < 200; i++ {
		pop = append(pop, ScoredLayer{newNetwork(), 0})
	}

	env := make([]byte, 9)
	for {
		for i := range pop {
			pop[i].Score = 0
		}

		for i := 0; i < 100; i++ {
			// Prepare environment and input.
			var n int
			for i := range env {
				n += 1
				if rand.Intn(2) == 0 && n < 9 {
					env[i] = 2
				} else {
					env[i] = 0
				}
			}
			copy(in, env)

			for j, p := range pop {
				values := p.GetValues()
				pop[j].Score += Step(in, values, env)
			}
		}

		// Find the highest scoring networks.
		sort.Slice(pop, func(i, j int) bool {
			return pop[i].Score > pop[j].Score
		})

		fmt.Printf("[%10d]", pop[0].Score)
		for _, v := range pop[0].GetValues() {
			fmt.Printf(" %3d", v)
		}
		fmt.Println()

		// 10 copies of the top network.
		for i := 10; i < 20; i++ {
			pop[i] = *pop[0].Copy().(*ScoredLayer)
			pop[i].Mutate(5000)
		}
		// 5 copies of 2nd and 3rd.
		for i := 20; i < 25; i++ {
			pop[i] = *pop[1].Copy().(*ScoredLayer)
			pop[i].Mutate(1000)
		}
		for i := 25; i < 30; i++ {
			pop[i] = *pop[2].Copy().(*ScoredLayer)
			pop[i].Mutate(500)
		}
		// Remaining bottom dies.
		for i := 30; i < len(pop); i++ {
			pop[i] = ScoredLayer{newNetwork(), 0}
		}
	}
}

func NewFullyConnectedLayer(left Layer, size int) *InferredLayer {
	l := &InferredLayer{
		Nodes: make([]Node, size),
		Left:  left,
	}
	leftSize := left.Size()
	r := make([]byte, len(l.Nodes)*leftSize*2)
	rand.Read(r)
	var ri int
	for i := 0; i < len(l.Nodes); i++ {
		edges := make([]Edge, leftSize)
		for j := 0; j < leftSize; j++ {
			edges[j].Index = j
			edges[j].And = r[ri]
			edges[j].Xor = r[ri+1]
			ri += 2
		}
		l.Nodes[i].Inputs = edges
	}
	return l
}

func Step(env1, out, env2 []byte) int {
	if len(env1) != len(out) || len(env1) != len(env2) {
		panic("length mismatch")
	}
	move := -1
	score := 0
	zeroes := 0
	for i, n := range out {
		if n == 0 {
			zeroes++
		} else if n == 1 {
			if move != -1 {
				// illegal move - only one per turn
				score -= 10
				continue
			}
			move = i
			score += 100
		} else {
			score -= 5 + int(n)
		}
	}
	score += zeroes * 7
	if zeroes == 8 && move != -1 && env1[move] == 0 {
		env2[move] = 1
		score += 100
	}
	return score + rand.Intn(10)
}
