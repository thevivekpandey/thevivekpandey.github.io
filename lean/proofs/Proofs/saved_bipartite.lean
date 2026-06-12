/-
Bipartite matching "rank" and the bound   rank G ≤ N (N - 1) / 2.

Encoding
--------
* The bipartite graph  G = (U, V, E)  with  |U| = |V| = N  is given by its edge
  relation  `E : Fin N → Fin N → Prop`  (the biadjacency relation:  `E i j`  means
  the i-th vertex of U is joined to the j-th vertex of V).
* A perfect matching is a permutation  `π : Equiv.Perm (Fin N)`  with  `E i (π i)`
  for every `i`.
* For a permutation `π`,        f(π, x, y) = ∑ᵢ xⁱ · y^(π i).
* For `n : ℕ`,                  g(n, x, y) = ∑_{π a matching} sign(π) · f(π, x, y)ⁿ.
* `rank G` is the least `n` with `g(n, ·, ·) ≠ 0`   (and `∞` if there is none).

The theorem is stated; its proof is left as `sorry`.
-/
import Mathlib

open MvPolynomial

namespace BipartiteRank

variable {N : ℕ}

/-- Bivariate integer polynomials `ℤ[x, y]`, with `x = X 0` and `y = X 1`. -/
abbrev Poly := MvPolynomial (Fin 2) ℤ

/-- Matching polynomial of a permutation `π`:   `f(π, x, y) = ∑ᵢ xⁱ · y^(π i)`. -/
noncomputable def f (π : Equiv.Perm (Fin N)) : Poly :=
  ∑ i : Fin N, X (0 : Fin 2) ^ (i : ℕ) * X (1 : Fin 2) ^ (π i : ℕ)

/-- `π` is a perfect matching of `G = (U, V, E)`: every matched pair is an edge. -/
abbrev IsPerfectMatching (E : Fin N → Fin N → Prop) (π : Equiv.Perm (Fin N)) : Prop :=
  ∀ i, E i (π i)

/-- The signed power sum
    `g(n, x, y) = ∑_{π a perfect matching} sign(π) · f(π, x, y)ⁿ`. -/
noncomputable def g (E : Fin N → Fin N → Prop) [DecidableRel E] (n : ℕ) : Poly :=
  ∑ π ∈ Finset.univ.filter (IsPerfectMatching E),
      (Equiv.Perm.sign π : ℤ) • f π ^ n

/-- `rank G` is the least `n` for which `g(n, ·, ·)` is not the zero polynomial,
    and `⊤ = ∞` when no such `n` exists — in particular when `G` has no perfect
    matching, in which case every `g(n, ·, ·)` is identically zero. -/
noncomputable def rank (E : Fin N → Fin N → Prop) [DecidableRel E] : ℕ∞ :=
  ⨅ n ∈ {m : ℕ | g E m ≠ 0}, (n : ℕ∞)

/-- **Theorem.**  Every bipartite graph `G = (U, V, E)` with `|U| = |V| = N`
    that admits at least one perfect matching satisfies
    `rank G ≤ N (N - 1) / 2`   ( = `Nat.choose N 2`). -/
theorem rank_le_choose_two
    (E : Fin N → Fin N → Prop) [DecidableRel E]
    (hpm : ∃ π : Equiv.Perm (Fin N), IsPerfectMatching E π) :
    rank E ≤ ((N * (N - 1) / 2 : ℕ) : ℕ∞) := by
  sorry

/-- The complete bipartite graph `K_{N,N}`: every vertex in U is connected to
    every vertex in V. -/
def IsCompleteBipartite (E : Fin N → Fin N → Prop) : Prop :=
  ∀ i j, E i j

/-- **Theorem.**  The complete bipartite graph `K_{N,N}` has rank exactly
    `N (N - 1) / 2`, achieving the upper bound. -/
theorem rank_complete
    (E : Fin N → Fin N → Prop) [DecidableRel E]
    (hcomplete : IsCompleteBipartite E) :
    rank E = ((N * (N - 1) / 2 : ℕ) : ℕ∞) := by
  -- The complete bipartite graph has all permutations as perfect matchings
  have all_matchings : ∀ π : Equiv.Perm (Fin N), IsPerfectMatching E π := by
    intro π i
    exact hcomplete i (π i)

  -- Strategy: Show that g E n ≠ 0 iff n ≥ N(N-1)/2
  -- This is a deep result requiring analysis of the signed sum
  -- The key insight is that for the complete graph, g E n is related to
  -- the permanent of a certain matrix, and the cancellations from the
  -- alternating signs cause g E n = 0 for n < N(N-1)/2.

  sorry

end BipartiteRank
