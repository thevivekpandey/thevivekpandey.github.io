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

/-- The minimum sum of N distinct natural numbers is 0+1+...+(N-1) = N(N-1)/2. -/
lemma sum_of_distinct_ge_triangular (N : ℕ) (c : Fin N → ℕ)
    (hdist : Function.Injective c) :
    ∑ i, c i ≥ N * (N - 1) / 2 := by
  let s := Finset.image c Finset.univ
  have hcard : s.card = N := by
    rw [Finset.card_image_of_injective Finset.univ hdist]
    simp
  have hsum : ∑ i, c i = ∑ x ∈ s, x := by
    rw [Finset.sum_image (fun _ _ _ _ h => hdist h)]
  rw [hsum]
  -- We leave the triangular inequality as sorry for now
  sorry

/-- The signed sum S(c) for a count vector c.
    S(c) = ∑_{π ∈ S_N} sign(π) · y^{∑ᵢ cᵢ·π(i)} -/
noncomputable def S (N : ℕ) (c : Fin N → ℕ) : Poly :=
  ∑ π : Equiv.Perm (Fin N),
    (Equiv.Perm.sign π : ℤ) • X (1 : Fin 2) ^ (∑ i, c i * (π i : ℕ))

/-- Key cancellation lemma: S(c) = 0 when c has duplicate values. -/
lemma S_eq_zero_of_duplicate {N : ℕ} (c : Fin N → ℕ)
    (hduplicate : ∃ a b : Fin N, a ≠ b ∧ c a = c b) :
    S N c = 0 := by
  obtain ⟨a, b, hab, hc⟩ := hduplicate
  let τ := Equiv.swap a b
  let φ : Equiv.Perm (Fin N) → Equiv.Perm (Fin N) := fun π => π * τ
  have hφ_sign : ∀ π, Equiv.Perm.sign (φ π) = -Equiv.Perm.sign π := by
    intro π
    unfold φ τ
    rw [Equiv.Perm.sign_mul, Equiv.Perm.sign_swap hab]
    simp
  have hφ_exp : ∀ π, ∑ i, c i * (φ π i : ℕ) = ∑ i, c i * (π i : ℕ) := by
    intro π
    have h_c_τ : ∀ i, c i = c (τ i) := by
      intro i
      by_cases hi : i = a
      · rw [hi, Equiv.swap_apply_left, hc]
      · by_cases hj : i = b
        · rw [hj, Equiv.swap_apply_right, ← hc]
        · unfold τ; rw [Equiv.swap_apply_of_ne_of_ne hi hj]
    have h_sum : ∑ i, c i * (φ π i : ℕ) = ∑ i, c (τ i) * (π (τ i) : ℕ) := by
      apply Finset.sum_congr rfl
      intro i _
      rw [h_c_τ i]
      rfl
    rw [h_sum]
    exact Equiv.sum_comp τ (fun i => c i * (π i : ℕ))
  unfold S
  apply Finset.sum_involution (fun π _ => φ π)
  · intro π _
    rw [hφ_sign π]
    simp [hφ_exp π]
  · intro π _ _ h_fixed
    have h_fixed_i : ∀ i, π (τ i) = π i := by
      intro i
      have h1 : φ π i = π i := congr_fun (congr_arg DFunLike.coe h_fixed) i
      exact h1
    have h_τ_a : τ a = a := π.injective (h_fixed_i a)
    have h_τ_a_2 : τ a = b := Equiv.swap_apply_left a b
    rw [h_τ_a_2] at h_τ_a
    exact hab h_τ_a.symm
  · intro π _
    exact Finset.mem_univ _
  · intro π _
    ext i
    have h_τ_τ : τ (τ i) = i := by
      by_cases hi : i = a
      · rw [hi]; unfold τ; simp
      · by_cases hj : i = b
        · rw [hj]; unfold τ; simp
        · unfold τ; rw [Equiv.swap_apply_of_ne_of_ne hi hj, Equiv.swap_apply_of_ne_of_ne hi hj]
    unfold φ
    simp only [Equiv.Perm.mul_apply]
    rw [h_τ_τ]

lemma f_pow_eq_sum_seq (π : Equiv.Perm (Fin N)) (n : ℕ) :
    f π ^ n = ∑ p : Fin n → Fin N, ∏ k, (X (0 : Fin 2) ^ (p k : ℕ) * X (1 : Fin 2) ^ (π (p k) : ℕ)) := by
  unfold f
  exact Finset.sum_pow (fun (i : Fin N) => X (0 : Fin 2) ^ (i : ℕ) * X (1 : Fin 2) ^ (π i : ℕ)) n

/-- **Theorem.**  The complete bipartite graph `K_{N,N}` has rank exactly
    `N (N - 1) / 2`, achieving the upper bound. -/
theorem rank_complete
    (E : Fin N → Fin N → Prop) [DecidableRel E]
    (hcomplete : IsCompleteBipartite E) :
    rank E = ((N * (N - 1) / 2 : ℕ) : ℕ∞) := by
  sorry

end BipartiteRank
