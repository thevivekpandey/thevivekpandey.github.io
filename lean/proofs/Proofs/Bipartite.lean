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
open Finset

namespace BipartiteRank

variable {N : ℕ}

/-- Bivariate integer polynomials `ℤ[x, y]`, with `x = X 0` and `y = X 1`. -/
abbrev Poly := MvPolynomial (Fin 2) ℤ

/-- Matching polynomial of a permutation `π`:   `f(π, x, y) = ∑ᵢ xⁱ · y^(π i)`. -/
noncomputable def f (π : Equiv.Perm (Fin N)) : Poly :=
  ∑ i : Fin N, X (0 : Fin 2) ^ (i : ℕ) * X (1 : Fin 2) ^ (π i : ℕ)

/-- The expansion of f(π)^n using the multinomial theorem.
    This connects to count vectors via Finset.piAntidiag.
    Key insight: piAntidiag n gives all functions c : Fin N → ℕ with ∑ᵢ c i = n -/
lemma f_pow_expansion (π : Equiv.Perm (Fin N)) (n : ℕ) :
    f π ^ n = ∑ c in (univ : Finset (Fin N)).piAntidiag n,
      (Nat.multinomial univ c : ℤ) •
      ∏ i : Fin N, (X (0 : Fin 2) ^ (i : ℕ) * X (1 : Fin 2) ^ (π i : ℕ)) ^ (c i) := by
  unfold f
  -- Apply the multinomial theorem
  rw [Finset.sum_pow_eq_sum_piAntidiag]
  -- The formula matches directly
  rfl

/-- Simplification: (xⁱ · y^{π(i)})^{c(i)} = x^{i·c(i)} · y^{π(i)·c(i)} -/
lemma monomial_power_simplify (π : Equiv.Perm (Fin N)) (i : Fin N) (c : ℕ) :
    (X (0 : Fin 2) ^ (i : ℕ) * X (1 : Fin 2) ^ (π i : ℕ)) ^ c =
    X (0 : Fin 2) ^ (i * c : ℕ) * X (1 : Fin 2) ^ ((π i : ℕ) * c) := by
  rw [mul_pow]
  congr 1 <;> rw [pow_mul]

/-- The x-degree (degree in variable 0) contributed by count vector c is ∑ᵢ i · c(i).
    This is the key quantity for the rearrangement inequality argument. -/
def xDegree (c : Fin N → ℕ) : ℕ := ∑ i, (i : ℕ) * c i

/-- Among all permutations of (0,1,...,N-1), the identity permutation id
    maximizes ∑ᵢ i · c(i). This follows from the rearrangement inequality:
    pairing the largest i with the largest c(i) gives the maximum sum. -/
lemma xDegree_id_maximal (c : Fin N → ℕ)
    (h_perm : ∃ σ : Equiv.Perm (Fin N), ∀ i, c i = σ i) :
    xDegree c ≤ xDegree id := by
  obtain ⟨σ, hσ⟩ := h_perm
  unfold xDegree
  -- c i = σ i, so ∑ i, i * c i = ∑ i, i * σ i
  conv_lhs => arg 2; ext i; rw [hσ i]
  -- Now apply rearrangement inequality
  -- Both i ↦ i and i ↦ σ i are monotone increasing on Fin N
  -- The sum ∑ i, i * σ i ≤ ∑ i, i * i by rearrangement
  -- This requires showing Monovary (id : Fin N → ℕ) (id : Fin N → ℕ)
  have h_mono : Monovary (fun i : Fin N => (i : ℕ)) (fun i : Fin N => (i : ℕ)) := by
    intro i j hij
    exact Fin.coe_fin_le.mpr hij
  exact Monovary.sum_mul_comp_perm_le_sum_mul h_mono σ

/-- The identity count vector id : Fin N → ℕ (where id i = i)
    gives x-degree equal to ∑ᵢ i² = 0² + 1² + ... + (N-1)². -/
lemma xDegree_id_formula : xDegree (id : Fin N → ℕ) = ∑ i : Fin N, (i : ℕ) ^ 2 := by
  unfold xDegree
  simp [id]
  congr 1
  ext i
  ring

/-- The signed sum S(c) = ∑_π sign(π) · y^{∑ᵢ c(i)·π(i)} for count vector c.
    This is the y-coefficient that arises in the count-vector expansion.
    When c = (0,1,...,N-1), this is related to the Vandermonde determinant. -/
noncomputable def S (c : Fin N → ℕ) : Poly :=
  ∑ π : Equiv.Perm (Fin N),
    (Equiv.Perm.sign π : ℤ) • X (1 : Fin 2) ^ (∑ i, c i * π i : ℕ)

/-- When the count vector c has all distinct values (i.e., is a permutation of 0,1,...,N-1),
    S(c) is nonzero. This is because it's related to a Vandermonde determinant.
    More precisely, when c is injective, S(c) is a constant multiple of the Vandermonde
    determinant det(y^{c(i)})_{i,j}, which is nonzero for distinct exponents. -/
lemma S_nonzero_of_injective (c : Fin N → ℕ) (h_inj : Function.Injective c) :
    S c ≠ 0 := by
  -- This requires showing the connection to Matrix.det_vandermonde_ne_zero_iff
  -- The key insight: S(c) can be rewritten as a determinant
  -- det(y^{c(0)}, y^{c(1)}, ..., y^{c(N-1)}) which is the Vandermonde determinant
  -- For distinct values c(i), this determinant is nonzero
  sorry

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
  -- Strategy: Show that g E n ≠ 0 for some n ≤ N(N-1)/2
  -- Then by definition of rank as infimum, rank E ≤ n ≤ N(N-1)/2

  -- Actually, we can use a more direct approach:
  -- For the identity permutation id, f(id) = x^0 + x^1 + ... + x^{N-1}
  -- At n = N(N-1)/2, the term from id in g E n will contribute something nonzero

  -- But even simpler: we know rank E exists (is not ⊤) because there's a perfect matching
  -- And rank E is the minimum n where g E n ≠ 0
  -- We just need to show such an n exists with n ≤ N(N-1)/2

  -- For now, we use the fact that for the complete bipartite graph (which has all permutations),
  -- rank_complete shows rank = N(N-1)/2
  -- For any other graph with fewer matchings, rank should be ≤ that

  -- This still requires the deep count-vector argument
  sorry

/-- A count vector: how many times each position is used in the expansion of f^n.
    `c i` is the number of times position i is selected. -/
def CountVec (N n : ℕ) := { c : Fin N → ℕ // ∑ i, c i = n }

/-- Count vector has a duplicate: two positions have the same count. -/
def CountVec.hasDuplicate {N n : ℕ} (c : CountVec N n) : Prop :=
  ∃ a b : Fin N, a ≠ b ∧ c.val a = c.val b

/-- Count vector has all distinct values. -/
def CountVec.allDistinct {N n : ℕ} (c : CountVec N n) : Prop :=
  Function.Injective c.val

/-- The minimum sum of N distinct natural numbers is 0+1+...+(N-1) = N(N-1)/2. -/
lemma sum_of_distinct_ge_triangular (N : ℕ) (c : Fin N → ℕ)
    (hdist : Function.Injective c) :
    ∑ i, c i ≥ N * (N - 1) / 2 := by
  -- The range of c contains N distinct natural numbers
  -- The smallest such set is {0, 1, 2, ..., N-1}
  -- whose sum is N*(N-1)/2

  -- First, show that the image of c has cardinality N
  have card_image : Finset.card (Finset.image c Finset.univ) = N := by
    rw [Finset.card_image_of_injective Finset.univ hdist]
    simp

  -- Any N distinct natural numbers have sum ≥ 0+1+...+(N-1) = N(N-1)/2
  -- This is because the smallest N natural numbers are 0,1,...,N-1

  -- The image is a finite set of N distinct natural numbers
  let S := Finset.image c Finset.univ

  -- The sum ∑ i, c i equals the sum of elements in the image S
  have sum_eq : ∑ i, c i = ∑ x ∈ S, x := by
    rw [← Finset.sum_image]
    · rfl
    · intros x _ y _ heq
      exact hdist heq

  rw [sum_eq]

  -- The minimum sum of N distinct natural numbers is achieved by {0,1,...,N-1}
  -- We'll show that S has at least N elements, and the sum of the N smallest naturals is N(N-1)/2

  -- S has N elements since c is injective
  have card_S : S.card = N := by
    rw [Finset.card_image_of_injective]
    · simp [Fintype.card_fin]
    · exact hdist

  -- Now we need: sum of N distinct naturals ≥ 0+1+...+(N-1)
  -- This is a standard result: the sum is minimized when S = {0,1,...,N-1}

  -- The sum ∑ x ∈ S, x is minimized when S is the first N naturals
  -- Any other choice would replace some small number k < N with a larger number ≥ N

  sorry  -- This genuinely requires a Mathlib lemma or manual proof about minimizing sums

/-- If the sum equals N(N-1)/2 and values are distinct, then c is a permutation of 0,1,...,N-1. -/
lemma distinct_sum_eq_triangular_iff_perm (N : ℕ) (c : Fin N → ℕ)
    (hdist : Function.Injective c) (hsum : ∑ i, c i = N * (N - 1) / 2) :
    ∃ σ : Equiv.Perm (Fin N), ∀ i, c i = σ i := by
  sorry

/-- The signed sum S(c) for a count vector c.
    S(c) = ∑_{π ∈ S_N} sign(π) · y^{∑ᵢ cᵢ·π(i)} -/
noncomputable def S (c : Fin N → ℕ) : Poly :=
  ∑ π : Equiv.Perm (Fin N),
    (Equiv.Perm.sign π : ℤ) • X (1 : Fin 2) ^ (∑ i, c i * (π i : ℕ))

/-- Given a permutation π and two positions a,b, construct π' that swaps outputs at a and b.
    π'(a) = π(b), π'(b) = π(a), π'(i) = π(i) otherwise. -/
def swapOutputs (π : Equiv.Perm (Fin N)) (a b : Fin N) : Equiv.Perm (Fin N) :=
  π * Equiv.swap (π a) (π b)

lemma swapOutputs_at_a (π : Equiv.Perm (Fin N)) (a b : Fin N) :
    swapOutputs π a b a = π b := by
  unfold swapOutputs
  simp [Equiv.Perm.mul_apply, Equiv.swap_apply_left]

lemma swapOutputs_at_b (π : Equiv.Perm (Fin N)) (a b : Fin N) :
    swapOutputs π a b b = π a := by
  unfold swapOutputs
  simp [Equiv.Perm.mul_apply, Equiv.swap_apply_right]

lemma swapOutputs_at_other (π : Equiv.Perm (Fin N)) (a b i : Fin N) (ha : i ≠ a) (hb : i ≠ b) :
    swapOutputs π a b i = π i := by
  unfold swapOutputs
  simp only [Equiv.Perm.mul_apply]
  apply congr_arg π
  apply Equiv.swap_apply_of_ne_of_ne
  · intro h
    have : π i = π a := h
    have : i = a := π.injective this
    exact ha this
  · intro h
    have : π i = π b := h
    have : i = b := π.injective this
    exact hb this

lemma sign_swapOutputs (π : Equiv.Perm (Fin N)) (a b : Fin N) (hab : a ≠ b) :
    Equiv.Perm.sign (swapOutputs π a b) = -Equiv.Perm.sign π := by
  unfold swapOutputs
  rw [Equiv.Perm.sign_mul]
  by_cases h : π a = π b
  · -- If π a = π b, then by injectivity a = b, contradicting hab
    have : a = b := π.injective h
    contradiction
  · rw [Equiv.Perm.sign_swap h]
    simp [mul_comm]

/-- The y-exponent ∑ᵢ c(i)·π(i) is preserved when swapping outputs at positions with equal counts. -/
lemma sum_eq_of_swap (c : Fin N → ℕ) (π : Equiv.Perm (Fin N)) (a b : Fin N) (hab : a ≠ b) (hc : c a = c b) :
    ∑ i, c i * (swapOutputs π a b i : ℕ) = ∑ i, c i * (π i : ℕ) := by
  -- Rewrite the sum by separating out positions a and b
  have sum_split : ∀ f : Fin N → ℕ, ∑ i, f i = f a + f b + ∑ i ∈ Finset.univ.erase a |>.erase b, f i := by
    intro f
    rw [← Finset.sum_erase_add _ _ (Finset.mem_univ a)]
    have hb : b ∈ (Finset.univ : Finset (Fin N)).erase a := by
      simp [hab.symm]
    rw [← Finset.sum_erase_add _ _ hb]
    ring

  rw [sum_split, sum_split (fun i => c i * (π i : ℕ))]

  -- Show the "rest" sums are equal
  have rest_eq : (∑ i ∈ Finset.univ.erase a |>.erase b, c i * (swapOutputs π a b i : ℕ)) =
                 (∑ i ∈ Finset.univ.erase a |>.erase b, c i * (π i : ℕ)) := by
    apply Finset.sum_congr rfl
    intro i hi
    simp at hi
    rw [swapOutputs_at_other π a b i hi.2 hi.1]

  rw [rest_eq, swapOutputs_at_a, swapOutputs_at_b]
  -- Now we have: c(a) * π(b) + c(b) * π(a) + rest = c(a) * π(a) + c(b) * π(b) + rest
  -- Substitute c(a) = c(b)
  rw [hc]
  ring

/-- Key cancellation lemma: S(c) = 0 when c has duplicate values.
    Proof: pair each permutation π with π' that swaps outputs at positions with equal counts.
    They contribute opposite signs but the same y-exponent, so they cancel. -/
lemma S_eq_zero_of_duplicate (c : Fin N → ℕ)
    (hduplicate : ∃ a b : Fin N, a ≠ b ∧ c a = c b) :
    S c = 0 := by
  obtain ⟨a, b, hab, hc⟩ := hduplicate

  unfold S

  -- Define the involution: π ↦ swapOutputs π a b
  let φ : Equiv.Perm (Fin N) → Equiv.Perm (Fin N) := fun π => swapOutputs π a b

  -- Key facts about the involution φ:
  -- 1. sign(φ π) = -sign(π)
  have sign_φ : ∀ π, Equiv.Perm.sign (φ π) = -Equiv.Perm.sign π := by
    intro π
    exact sign_swapOutputs π a b hab

  -- 2. The y-exponent is preserved: ∑ᵢ c(i)·(φ π)(i) = ∑ᵢ c(i)·π(i)
  have exponent_φ : ∀ π, ∑ i, c i * (φ π i : ℕ) = ∑ i, c i * (π i : ℕ) := by
    intro π
    exact sum_eq_of_swap c π a b hab hc

  -- Now we show the sum equals zero by the pairing argument
  -- Each term for π cancels with the term for φ(π)

  -- We'll use a manual involution argument
  -- Pair each π with φ π. They have opposite signs but same exponent, so cancel

  -- First, show φ is its own inverse
  have φ_involution : ∀ π, φ (φ π) = π := by
    intro π
    unfold φ swapOutputs
    -- swapOutputs π a b = π * swap(π a, π b)
    -- So φ(φ π) = (π * swap) * swap(π'a, π'b) where π' = π * swap
    -- We need: π'a = (π*swap)(a) = π(swap a) and similarly for b
    ext i
    simp only [Equiv.Perm.coe_mul]
    -- The key: swap(π a, π b) applied twice at any point gives identity
    by_cases h : π i = π a ∨ π i = π b
    · cases h with
      | inl ha =>
        -- π i = π a
        simp [Equiv.swap_apply_left, ha]
      | inr hb =>
        -- π i = π b
        simp [Equiv.swap_apply_right, hb]
    · push_neg at h
      obtain ⟨hna, hnb⟩ := h
      simp [Equiv.swap_apply_of_ne_of_ne hna hnb]

  -- Now use that φ partitions the permutations into pairs that cancel
  -- The sum is: ∑_π sign(π) · y^{∑ᵢ cᵢ·π(i)}
  -- Group into pairs {π, φ(π)}. For each pair:
  --   sign(π) · y^{exp} + sign(φ π) · y^{exp} = sign(π) · y^{exp} + (-sign(π)) · y^{exp} = 0

  -- Split the sum based on a partition into pairs
  -- Use the fact that the sum over pairs {π, φ(π)} cancels

  -- The actual proof would use Finset.sum_involution or a manual pairing argument
  -- Key insight: for each π, we have:
  --   term(π) + term(φ π) = sign(π)·y^exp + sign(φ π)·y^exp
  --                       = sign(π)·y^exp + (-sign(π))·y^exp    (by sign_φ)
  --                       = 0
  -- Since φ is an involution (φ∘φ = id), every π is paired with exactly one other element
  -- (or is a fixed point, but we'd need to show φ has no fixed points given hab)

  -- Show φ has no fixed points
  have no_fixed_points : ∀ π, φ π ≠ π := by
    intro π h_fixed
    -- If φ π = π, then swapOutputs π a b = π
    -- This means π * swap(π a, π b) = π
    -- So swap(π a, π b) = id
    -- But swap(π a, π b) = id only if π a = π b
    -- This contradicts the fact that π is injective and a ≠ b
    unfold φ swapOutputs at h_fixed
    have : Equiv.swap (π a) (π b) = 1 := by
      have : π * Equiv.swap (π a) (π b) = π * 1 := by rw [h_fixed]; simp
      exact mul_left_cancel this
    have : π a = π b := Equiv.swap_eq_one_iff.mp this
    have : a = b := π.injective this
    exact hab this

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

  -- We need to show: rank E = N(N-1)/2
  -- This means: g E n = 0 for n < N(N-1)/2, and g E (N(N-1)/2) ≠ 0

  -- Step 1: For the complete graph, g E n can be rewritten in terms of count vectors
  -- g(n) = ∑_c (n!/∏ᵢcᵢ!) · x^(∑ᵢ i·cᵢ) · S(c)
  -- where the sum is over count vectors c with ∑cᵢ = n

  -- Step 2: S(c) = 0 when c has duplicate values (key cancellation lemma)
  -- So only count vectors with all distinct values survive

  -- Step 3: For n < N(N-1)/2, there exists no count vector with:
  --   * ∑cᵢ = n
  --   * all cᵢ distinct
  -- because the minimum sum of N distinct non-negative integers is 0+1+...+(N-1) = N(N-1)/2
  have g_zero_below : ∀ n < N * (N - 1) / 2, g E n = 0 := by
    intro n hn
    -- Strategy: Show that g E n, when expanded, is a sum over count vectors.
    -- Each count vector c contributes a term proportional to S(c).
    -- By S_eq_zero_of_duplicate, any c with duplicates contributes 0.
    -- For c with all distinct values, by sum_of_distinct_ge_triangular,
    -- we have ∑ᵢ c i ≥ N(N-1)/2 > n, so no such c exists with sum = n.
    -- Therefore all terms vanish and g E n = 0.

    -- The expansion of (∑ᵢ xⁱ·y^{π(i)})ⁿ gives terms indexed by count vectors
    -- This requires showing that g E n can be written as:
    -- g E n = ∑_c (multinomial coefficient) · x^(something) · S(c)
    -- where c ranges over count vectors with ∑ᵢ c i = n

    -- This is the fundamental count-vector expansion of the signed power sum,
    -- but proving it rigorously requires significant combinatorial work
    sorry

  -- Step 4: At n = N(N-1)/2, the count vector (0,1,...,N-1) (and its permutations) survive
  -- The highest x-power comes uniquely from c = (0,1,...,N-1) by rearrangement inequality
  -- Its coefficient is nonzero (related to Vandermonde determinant)
  have g_nonzero_at : g E (N * (N - 1) / 2) ≠ 0 := by
    -- Strategy: Show g has a nonzero coefficient for some monomial
    -- Use MvPolynomial.ne_zero_iff: p ≠ 0 ↔ ∃ d, coeff d p ≠ 0

    -- Step 4a: The identity count vector c_id : Fin N → ℕ where c_id i = i
    -- sums to N(N-1)/2
    let c_id : Fin N → ℕ := fun i => i
    have sum_c_id : ∑ i, c_id i = N * (N - 1) / 2 := by
      simp only [c_id]
      -- This is the triangular number formula: ∑ i < N, i = N(N-1)/2
      -- Mathlib has Finset.sum_range_id : ∑ i ∈ range n, i = n * (n - 1) / 2
      convert Finset.sum_range_id N using 1
      · ext i
        simp [Finset.mem_range]
      · ring

    -- Step 4b: The x-degree of c_id is ∑ᵢ i² (by xDegree_id_formula)
    have xdeg_id : xDegree c_id = ∑ i : Fin N, (i : ℕ) ^ 2 := by
      exact xDegree_id_formula

    -- Step 4c: Among all permutations of (0,1,...,N-1), c_id gives maximal x-degree
    have xdeg_maximal : ∀ σ : Equiv.Perm (Fin N),
        xDegree (fun i => σ i) ≤ xDegree c_id := by
      intro σ
      apply xDegree_id_maximal
      use σ
      intro i
      rfl

    -- Step 4d: The key claim - there exists a monomial with nonzero coefficient
    -- This requires:
    --   1. Expanding g using f_pow_expansion
    --   2. Showing the count vector c_id contributes a unique leading term
    --   3. Computing that term's coefficient as a Vandermonde-type sum
    --   4. Showing the Vandermonde sum is nonzero

    -- For now, we assert this key fact:
    have exists_nonzero_coeff : ∃ (m : Fin 2 →₀ ℕ), (g E (N * (N - 1) / 2)).coeff m ≠ 0 := by
      -- Strategy: We'll show that g has a term with x-degree = ∑ᵢ i²
      -- whose coefficient is a nonzero Vandermonde-type sum

      -- Step 1: Expand g using the definition
      unfold g

      -- g E n = ∑ π ∈ (perfect matchings), sign(π) • f(π)^n
      -- Each f(π)^n expands via f_pow_expansion into count vectors

      -- Step 2: The key observation is that at n = N(N-1)/2,
      -- the only count vectors that sum to n with all distinct values
      -- are permutations of (0,1,...,N-1)

      -- Step 3: Among these, c_id = (0,1,...,N-1) gives the unique maximum x-degree
      -- by xdeg_maximal

      -- Step 4: For this leading x-degree, we need to compute the y-coefficient
      -- This involves summing over all permutations π with sign(π)

      -- The full expansion requires:
      -- 1. Using f_pow_expansion to rewrite each f(π)^n
      -- 2. Reorganizing the double sum (over π and count vectors)
      -- 3. Showing cancellation for non-distinct count vectors
      -- 4. Computing the surviving coefficient as a Vandermonde determinant
      -- 5. Using Matrix.det_vandermonde_ne_zero_iff to show nonzero

      -- This is substantial work connecting polynomial theory to combinatorics
      sorry

    -- Step 4e: If there exists a nonzero coefficient, the polynomial is nonzero
    exact MvPolynomial.ne_zero_iff.mpr exists_nonzero_coeff

  -- Step 5: Combine to conclude rank E = N(N-1)/2
  -- rank E is the infimum of {n | g E n ≠ 0}
  -- We've shown:
  --   * For n < N(N-1)/2: g E n = 0 (so n ∉ {n | g E n ≠ 0})
  --   * For n = N(N-1)/2: g E n ≠ 0 (so N(N-1)/2 ∈ {n | g E n ≠ 0})
  -- Therefore N(N-1)/2 is the minimum element of {n | g E n ≠ 0}
  -- So rank E = N(N-1)/2

  -- First show that N(N-1)/2 ∈ {n | g E n ≠ 0}
  have mem_set : (N * (N - 1) / 2 : ℕ∞) ∈ {(n : ℕ∞) | g E n ≠ 0} := by
    simp
    exact g_nonzero_at

  -- The infimum of a nonempty set of naturals that contains k is at most k
  have rank_le : rank E ≤ N * (N - 1) / 2 := by
    unfold rank
    apply csInf_le
    · use 0
      intro n hn
      simp at hn
    · exact mem_set

  -- For the other direction, show that for all n < N(N-1)/2, n ∉ {n | g E n ≠ 0}
  have rank_ge : (N * (N - 1) / 2 : ℕ∞) ≤ rank E := by
    unfold rank
    apply le_csInf
    · use N * (N - 1) / 2
      simp
      exact g_nonzero_at
    · intro m hm
      simp at hm
      -- hm : g E m ≠ 0
      -- Need to show: N(N-1)/2 ≤ m
      by_contra h_neg
      push_neg at h_neg
      -- h_neg : m < N(N-1)/2
      have : g E m = 0 := g_zero_below m h_neg
      contradiction

  exact le_antisymm rank_le rank_ge

end BipartiteRank
