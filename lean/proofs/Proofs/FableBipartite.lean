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

Proof sketch for `rank_complete`
--------------------------------
For the complete graph every permutation is a matching.  Expanding `f(π)ⁿ` by
the multinomial theorem and swapping the two sums,

  g(n) = ∑_{c : Fin N → ℕ, ∑ c = n} multinomial(c) · x^(∑ i·cᵢ) · D(c),
  D(c) = ∑_π sign(π) · y^(∑ cᵢ·π(i)) = det ((y^cᵢ)^j)ᵢⱼ = ∏_{i<j} (y^cⱼ - y^cᵢ),

a (generalized) Vandermonde determinant.  If `c` is not injective then `D(c) = 0`;
an injective `c` has `∑ c ≥ 0 + 1 + ⋯ + (N-1) = N(N-1)/2`, so `g(n) = 0` for all
`n < N(N-1)/2`.  At `n = N(N-1)/2` the surviving `c` are exactly the permutations,
all carrying the same multinomial coefficient `C > 0`, and

  g(N(N-1)/2) = C · ∏_{i<j} (x^j - x^i) · ∏_{i<j} (y^j - y^i) ≠ 0

in the integral domain ℤ[x,y].  Hence `rank = N(N-1)/2`.
-/
import Mathlib

open MvPolynomial Finset

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

/-- The complete bipartite graph `K_{N,N}`: every vertex in U is connected to
    every vertex in V. -/
def IsCompleteBipartite (E : Fin N → Fin N → Prop) : Prop :=
  ∀ i j, E i j

/-! ### Sums of distinct natural numbers -/

/-- A finset of naturals either is an initial segment `{0, …, card-1}` or its sum
    strictly exceeds `0 + 1 + ⋯ + (card-1)`. -/
private lemma eq_range_card_or_sum_lt (S : Finset ℕ) :
    S = Finset.range S.card ∨ ∑ j ∈ Finset.range S.card, j < ∑ a ∈ S, a := by
  induction S using Finset.strongInduction with
  | _ S ih =>
    rcases S.eq_empty_or_nonempty with rfl | hS
    · left; simp
    · have hm : S.max' hS ∈ S := S.max'_mem hS
      set m := S.max' hS with hmdef
      obtain ⟨c, hc⟩ : ∃ c, (S.erase m).card = c := ⟨_, rfl⟩
      have hins : S = insert m (S.erase m) := (Finset.insert_erase hm).symm
      have hcard : S.card = c + 1 := by
        rw [← hc, Finset.card_erase_of_mem hm]
        exact (Nat.succ_pred_eq_of_pos (Finset.card_pos.mpr hS)).symm
      have hm_ge : c ≤ m := by
        have hsub : S ⊆ Finset.range (m + 1) := fun a ha =>
          Finset.mem_range.mpr (Nat.lt_succ_of_le (S.le_max' a ha))
        have h1 := Finset.card_le_card hsub
        rw [Finset.card_range] at h1
        omega
      have hsum : ∑ a ∈ S, a = m + ∑ a ∈ S.erase m, a := by
        conv_lhs => rw [hins]
        exact Finset.sum_insert (Finset.notMem_erase m S)
      rcases ih (S.erase m) (Finset.erase_ssubset hm) with h' | h'
      · rw [hc] at h'
        rcases eq_or_lt_of_le hm_ge with hmn | hmn
        · left
          calc S = insert m (S.erase m) := hins
            _ = insert m (Finset.range c) := by rw [h']
            _ = insert c (Finset.range c) := by rw [hmn]
            _ = Finset.range (c + 1) := (Finset.range_add_one).symm
            _ = Finset.range S.card := by rw [hcard]
        · right
          rw [hcard, Finset.sum_range_succ, hsum, h']
          omega
      · right
        rw [hc] at h'
        rw [hcard, Finset.sum_range_succ, hsum]
        omega

/-- `N` distinct naturals sum to at least `0 + 1 + ⋯ + (N-1)`. -/
private lemma triangle_le_sum_of_injective {k : Fin N → ℕ} (hk : Function.Injective k) :
    ∑ i : Fin N, (i : ℕ) ≤ ∑ i : Fin N, k i := by
  have himage : ∑ a ∈ Finset.image k Finset.univ, a = ∑ i : Fin N, k i :=
    Finset.sum_image (fun i _ j _ h => hk h)
  have hcard : (Finset.image k Finset.univ).card = N := by
    rw [Finset.card_image_of_injective _ hk, Finset.card_univ, Fintype.card_fin]
  have hle : ∑ j ∈ Finset.range N, j ≤ ∑ a ∈ Finset.image k Finset.univ, a := by
    rcases eq_range_card_or_sum_lt (Finset.image k Finset.univ) with h | h
    · rw [h, hcard]
    · rw [hcard] at h; exact h.le
  calc ∑ i : Fin N, (i : ℕ) = ∑ j ∈ Finset.range N, j :=
        Fin.sum_univ_eq_sum_range (fun j => j) N
    _ ≤ ∑ a ∈ Finset.image k Finset.univ, a := hle
    _ = ∑ i : Fin N, k i := himage

/-- Equality case: `N` distinct naturals summing to exactly `0 + 1 + ⋯ + (N-1)`
    are a permutation of `{0, …, N-1}`. -/
private lemma exists_perm_of_injective_of_sum_eq {k : Fin N → ℕ}
    (hk : Function.Injective k) (hsum : ∑ i : Fin N, k i = ∑ i : Fin N, (i : ℕ)) :
    ∃ σ : Equiv.Perm (Fin N), ∀ i, k i = (σ i : ℕ) := by
  have hcard : (Finset.image k Finset.univ).card = N := by
    rw [Finset.card_image_of_injective _ hk, Finset.card_univ, Fintype.card_fin]
  have himage : ∑ a ∈ Finset.image k Finset.univ, a = ∑ i : Fin N, k i :=
    Finset.sum_image (fun i _ j _ h => hk h)
  have hrange : Finset.image k Finset.univ = Finset.range N := by
    rcases eq_range_card_or_sum_lt (Finset.image k Finset.univ) with h | h
    · rw [hcard] at h; exact h
    · exfalso
      rw [hcard, himage, hsum] at h
      rw [Fin.sum_univ_eq_sum_range (fun j => j) N] at h
      exact lt_irrefl _ h
  have hlt : ∀ i, k i < N := fun i => by
    have hmem : k i ∈ Finset.image k Finset.univ :=
      Finset.mem_image_of_mem k (Finset.mem_univ i)
    rw [hrange] at hmem
    exact Finset.mem_range.mp hmem
  have hσ₀ : Function.Injective (fun i => (⟨k i, hlt i⟩ : Fin N)) := fun i j h =>
    hk (congrArg Fin.val h)
  exact ⟨Equiv.ofBijective _ ((Finite.injective_iff_bijective).mp hσ₀), fun i => rfl⟩

/-! ### The alternating sum as a Vandermonde determinant -/

/-- `∑_π sign(π) · ∏ᵢ wᵢ^{π(i)}` is the determinant of the (generalized)
    Vandermonde matrix `(wᵢʲ)ᵢⱼ`. -/
private lemma altSum_eq_det (w : Fin N → Poly) :
    ∑ π : Equiv.Perm (Fin N), ((Equiv.Perm.sign π : ℤ) : Poly) * ∏ i, w i ^ (π i : ℕ)
      = (Matrix.vandermonde w).det := by
  rw [← Matrix.det_transpose, Matrix.det_apply']
  refine Finset.sum_congr rfl fun π _ => ?_
  congr 1

/-- The Vandermonde determinant in powers of `y` vanishes when the exponents
    repeat. -/
private lemma det_vandermonde_eq_zero_of_not_injective {k : Fin N → ℕ}
    (hk : ¬ Function.Injective k) :
    (Matrix.vandermonde fun i : Fin N => (X (1 : Fin 2) : Poly) ^ k i).det = 0 := by
  rw [Matrix.det_vandermonde]
  rw [Function.Injective] at hk
  push_neg at hk
  obtain ⟨a, b, hab, hne⟩ := hk
  rcases hne.lt_or_gt with h | h
  · exact Finset.prod_eq_zero (Finset.mem_univ a)
      (Finset.prod_eq_zero (Finset.mem_Ioi.mpr h) (by rw [hab, sub_self]))
  · exact Finset.prod_eq_zero (Finset.mem_univ b)
      (Finset.prod_eq_zero (Finset.mem_Ioi.mpr h) (by rw [hab, sub_self]))

/-- The Vandermonde determinant in distinct powers of a variable is nonzero. -/
private lemma det_vandermonde_pow_ne_zero (v : Fin 2) :
    (Matrix.vandermonde fun i : Fin N => (X v : Poly) ^ (i : ℕ)).det ≠ 0 := by
  rw [Matrix.det_vandermonde]
  refine Finset.prod_ne_zero_iff.mpr fun i _ => Finset.prod_ne_zero_iff.mpr fun j hj => ?_
  rw [sub_ne_zero]
  intro hXX
  rw [MvPolynomial.X_pow_eq_monomial, MvPolynomial.X_pow_eq_monomial] at hXX
  have h1 := MvPolynomial.monomial_left_injective (R := ℤ) one_ne_zero hXX
  have h2 := Finsupp.single_injective v h1
  have h3 : i < j := Finset.mem_Ioi.mp hj
  omega

/-! ### Expansion of `g` -/

/-- Multinomial expansion of `g` for the complete graph:
    `g(n) = ∑_{c, ∑c = n} multinomial(c) · x^(∑ i·cᵢ) · det((y^cᵢ)ʲ)`. -/
private lemma g_expand (E : Fin N → Fin N → Prop) [DecidableRel E]
    (hcomplete : IsCompleteBipartite E) (n : ℕ) :
    g E n = ∑ k ∈ Finset.piAntidiag (Finset.univ : Finset (Fin N)) n,
      (Nat.multinomial Finset.univ k : Poly) *
        (X (0 : Fin 2) ^ (∑ i : Fin N, (i : ℕ) * k i) *
          (Matrix.vandermonde fun i : Fin N => (X (1 : Fin 2) : Poly) ^ k i).det) := by
  have hfilter : Finset.univ.filter (IsPerfectMatching E) = Finset.univ :=
    Finset.filter_true_of_mem fun π _ i => hcomplete i (π i)
  rw [g, hfilter]
  have hfpow : ∀ π : Equiv.Perm (Fin N),
      f π ^ n = ∑ k ∈ Finset.piAntidiag (Finset.univ : Finset (Fin N)) n,
        (Nat.multinomial Finset.univ k : Poly) *
          (X (0 : Fin 2) ^ (∑ i : Fin N, (i : ℕ) * k i) *
            ∏ i : Fin N, ((X (1 : Fin 2) : Poly) ^ k i) ^ (π i : ℕ)) := by
    intro π
    rw [f, Finset.sum_pow_eq_sum_piAntidiag]
    refine Finset.sum_congr rfl fun k _ => ?_
    congr 1
    calc ∏ i : Fin N, (X (0 : Fin 2) ^ (i : ℕ) * X (1 : Fin 2) ^ (π i : ℕ)) ^ k i
        = ∏ i : Fin N,
            ((X (0 : Fin 2) : Poly) ^ (i : ℕ)) ^ k i * (X (1 : Fin 2) ^ (π i : ℕ)) ^ k i :=
          Finset.prod_congr rfl fun i _ => mul_pow _ _ _
      _ = (∏ i : Fin N, ((X (0 : Fin 2) : Poly) ^ (i : ℕ)) ^ k i) *
            ∏ i : Fin N, ((X (1 : Fin 2) : Poly) ^ (π i : ℕ)) ^ k i :=
          Finset.prod_mul_distrib
      _ = X (0 : Fin 2) ^ (∑ i : Fin N, (i : ℕ) * k i) *
            ∏ i : Fin N, ((X (1 : Fin 2) : Poly) ^ k i) ^ (π i : ℕ) := by
          congr 1
          · rw [← Finset.prod_pow_eq_pow_sum]
            exact Finset.prod_congr rfl fun i _ => (pow_mul _ _ _).symm
          · exact Finset.prod_congr rfl fun i _ => by
              rw [← pow_mul, ← pow_mul, mul_comm]
  calc ∑ π : Equiv.Perm (Fin N), (Equiv.Perm.sign π : ℤ) • f π ^ n
      = ∑ π : Equiv.Perm (Fin N),
          ∑ k ∈ Finset.piAntidiag (Finset.univ : Finset (Fin N)) n,
            ((Equiv.Perm.sign π : ℤ) : Poly) *
              ((Nat.multinomial Finset.univ k : Poly) *
                (X (0 : Fin 2) ^ (∑ i : Fin N, (i : ℕ) * k i) *
                  ∏ i : Fin N, ((X (1 : Fin 2) : Poly) ^ k i) ^ (π i : ℕ))) := by
        refine Finset.sum_congr rfl fun π _ => ?_
        rw [hfpow π, Finset.smul_sum]
        exact Finset.sum_congr rfl fun k _ => zsmul_eq_mul _ _
    _ = ∑ k ∈ Finset.piAntidiag (Finset.univ : Finset (Fin N)) n,
          ∑ π : Equiv.Perm (Fin N),
            ((Equiv.Perm.sign π : ℤ) : Poly) *
              ((Nat.multinomial Finset.univ k : Poly) *
                (X (0 : Fin 2) ^ (∑ i : Fin N, (i : ℕ) * k i) *
                  ∏ i : Fin N, ((X (1 : Fin 2) : Poly) ^ k i) ^ (π i : ℕ))) :=
        Finset.sum_comm
    _ = ∑ k ∈ Finset.piAntidiag (Finset.univ : Finset (Fin N)) n,
          (Nat.multinomial Finset.univ k : Poly) *
            (X (0 : Fin 2) ^ (∑ i : Fin N, (i : ℕ) * k i) *
              ∑ π : Equiv.Perm (Fin N),
                ((Equiv.Perm.sign π : ℤ) : Poly) *
                  ∏ i : Fin N, ((X (1 : Fin 2) : Poly) ^ k i) ^ (π i : ℕ)) := by
        refine Finset.sum_congr rfl fun k _ => ?_
        simp only [Finset.mul_sum]
        exact Finset.sum_congr rfl fun π _ => by ring
    _ = ∑ k ∈ Finset.piAntidiag (Finset.univ : Finset (Fin N)) n,
          (Nat.multinomial Finset.univ k : Poly) *
            (X (0 : Fin 2) ^ (∑ i : Fin N, (i : ℕ) * k i) *
              (Matrix.vandermonde fun i : Fin N => (X (1 : Fin 2) : Poly) ^ k i).det) := by
        refine Finset.sum_congr rfl fun k _ => ?_
        rw [altSum_eq_det]

/-! ### Vanishing below the threshold -/

/-- `g(n) = 0` for `n < 0 + 1 + ⋯ + (N-1)`. -/
private lemma g_eq_zero_of_lt (E : Fin N → Fin N → Prop) [DecidableRel E]
    (hcomplete : IsCompleteBipartite E) {n : ℕ} (hn : n < ∑ i : Fin N, (i : ℕ)) :
    g E n = 0 := by
  rw [g_expand E hcomplete n]
  refine Finset.sum_eq_zero fun k hk => ?_
  have hsum : ∑ i : Fin N, k i = n := (Finset.mem_piAntidiag.mp hk).1
  have hkni : ¬ Function.Injective k := by
    intro hinj
    have hge := triangle_le_sum_of_injective hinj
    rw [hsum] at hge
    omega
  rw [det_vandermonde_eq_zero_of_not_injective hkni, mul_zero, mul_zero]

/-! ### Nonvanishing at the threshold -/

/-- `g(0 + 1 + ⋯ + (N-1)) ≠ 0`. -/
private lemma g_triangle_ne_zero (E : Fin N → Fin N → Prop) [DecidableRel E]
    (hcomplete : IsCompleteBipartite E) :
    g E (∑ i : Fin N, (i : ℕ)) ≠ 0 := by
  classical
  -- the summand of the expansion, as a function of the exponent vector `k`
  set T : (Fin N → ℕ) → Poly := fun k =>
    (Nat.multinomial Finset.univ k : Poly) *
      (X (0 : Fin 2) ^ (∑ i : Fin N, (i : ℕ) * k i) *
        (Matrix.vandermonde fun i : Fin N => (X (1 : Fin 2) : Poly) ^ k i).det) with hT
  -- the exponent vectors coming from permutations
  have hinj : Function.Injective
      (fun (σ : Equiv.Perm (Fin N)) => fun (i : Fin N) => ((σ i : ℕ))) := by
    intro σ τ h
    apply Equiv.ext
    intro i
    exact Fin.val_injective (congrFun h i)
  have hBA : Finset.image (fun (σ : Equiv.Perm (Fin N)) => fun (i : Fin N) => ((σ i : ℕ)))
      Finset.univ ⊆ Finset.piAntidiag (Finset.univ : Finset (Fin N)) (∑ i : Fin N, (i : ℕ)) := by
    intro k hk
    obtain ⟨σ, -, rfl⟩ := Finset.mem_image.mp hk
    rw [Finset.mem_piAntidiag]
    exact ⟨Equiv.sum_comp σ (fun i : Fin N => (i : ℕ)), fun i _ => Finset.mem_univ i⟩
  -- terms outside the image of the permutations vanish
  have hzero : ∀ k ∈ Finset.piAntidiag (Finset.univ : Finset (Fin N)) (∑ i : Fin N, (i : ℕ)),
      k ∉ Finset.image (fun (σ : Equiv.Perm (Fin N)) => fun (i : Fin N) => ((σ i : ℕ)))
        Finset.univ → T k = 0 := by
    intro k hkA hkB
    have hkni : ¬ Function.Injective k := by
      intro hinj'
      obtain ⟨σ, hσ⟩ := exists_perm_of_injective_of_sum_eq hinj'
        (Finset.mem_piAntidiag.mp hkA).1
      exact hkB (Finset.mem_image.mpr
        ⟨σ, Finset.mem_univ σ, (funext fun i => (hσ i).symm)⟩)
    rw [hT]
    simp only
    rw [det_vandermonde_eq_zero_of_not_injective hkni, mul_zero, mul_zero]
  -- reduce the sum to a sum over permutations
  have hsum_eq : g E (∑ i : Fin N, (i : ℕ)) =
      ∑ σ : Equiv.Perm (Fin N), T (fun i => ((σ i : ℕ))) := by
    rw [g_expand E hcomplete _, ← Finset.sum_subset hBA hzero,
      Finset.sum_image fun σ _ τ _ h => hinj h]
  -- the multinomial coefficient is the same for every permutation
  have hmult : ∀ σ : Equiv.Perm (Fin N),
      Nat.multinomial Finset.univ (fun i => ((σ i : ℕ))) =
        Nat.multinomial Finset.univ (fun i : Fin N => (i : ℕ)) := by
    intro σ
    have hs : ∑ i : Fin N, ((σ i : ℕ)) = ∑ i : Fin N, (i : ℕ) :=
      Equiv.sum_comp σ (fun i : Fin N => (i : ℕ))
    have hp : ∏ i : Fin N, ((σ i : ℕ)).factorial = ∏ i : Fin N, ((i : ℕ)).factorial :=
      Equiv.prod_comp σ (fun i : Fin N => ((i : ℕ)).factorial)
    simp only [Nat.multinomial]
    rw [hs, hp]
  -- the determinant picks up the sign of the permutation
  have hdet : ∀ σ : Equiv.Perm (Fin N),
      (Matrix.vandermonde fun i : Fin N => (X (1 : Fin 2) : Poly) ^ ((σ i : ℕ))).det =
        ((Equiv.Perm.sign σ : ℤ) : Poly) *
          (Matrix.vandermonde fun i : Fin N => (X (1 : Fin 2) : Poly) ^ (i : ℕ)).det := by
    intro σ
    have hsub : (Matrix.vandermonde fun i : Fin N => (X (1 : Fin 2) : Poly) ^ ((σ i : ℕ))) =
        (Matrix.vandermonde fun i : Fin N => (X (1 : Fin 2) : Poly) ^ (i : ℕ)).submatrix σ id :=
      by
        ext i j
        simp [Matrix.vandermonde_apply, Matrix.submatrix_apply]
    rw [hsub, Matrix.det_permute]
  -- the signed sum over permutations of the `x`-monomials is again Vandermonde
  have hVx : ∑ σ : Equiv.Perm (Fin N),
      ((Equiv.Perm.sign σ : ℤ) : Poly) * X (0 : Fin 2) ^ (∑ i : Fin N, (i : ℕ) * ((σ i : ℕ)))
        = (Matrix.vandermonde fun i : Fin N => (X (0 : Fin 2) : Poly) ^ (i : ℕ)).det := by
    rw [← altSum_eq_det]
    refine Finset.sum_congr rfl fun σ _ => ?_
    congr 1
    rw [← Finset.prod_pow_eq_pow_sum]
    exact Finset.prod_congr rfl fun i _ => pow_mul _ _ _
  -- assemble:  g(n₀) = C · V(x) · V(y)
  have hfinal : g E (∑ i : Fin N, (i : ℕ)) =
      (Nat.multinomial Finset.univ (fun i : Fin N => (i : ℕ)) : Poly) *
        ((Matrix.vandermonde fun i : Fin N => (X (0 : Fin 2) : Poly) ^ (i : ℕ)).det *
          (Matrix.vandermonde fun i : Fin N => (X (1 : Fin 2) : Poly) ^ (i : ℕ)).det) := by
    rw [hsum_eq]
    calc ∑ σ : Equiv.Perm (Fin N), T (fun i => ((σ i : ℕ)))
        = ∑ σ : Equiv.Perm (Fin N),
            (Nat.multinomial Finset.univ (fun i : Fin N => (i : ℕ)) : Poly) *
              (X (0 : Fin 2) ^ (∑ i : Fin N, (i : ℕ) * ((σ i : ℕ))) *
                (((Equiv.Perm.sign σ : ℤ) : Poly) *
                  (Matrix.vandermonde
                    fun i : Fin N => (X (1 : Fin 2) : Poly) ^ (i : ℕ)).det)) := by
          refine Finset.sum_congr rfl fun σ _ => ?_
          rw [hT]
          simp only
          rw [hmult σ, hdet σ]
      _ = (Nat.multinomial Finset.univ (fun i : Fin N => (i : ℕ)) : Poly) *
            ((∑ σ : Equiv.Perm (Fin N),
              ((Equiv.Perm.sign σ : ℤ) : Poly) *
                X (0 : Fin 2) ^ (∑ i : Fin N, (i : ℕ) * ((σ i : ℕ)))) *
              (Matrix.vandermonde fun i : Fin N => (X (1 : Fin 2) : Poly) ^ (i : ℕ)).det) := by
          simp only [Finset.mul_sum, Finset.sum_mul]
          exact Finset.sum_congr rfl fun σ _ => by ring
      _ = _ := by rw [hVx]
  rw [hfinal]
  refine mul_ne_zero ?_ (mul_ne_zero (det_vandermonde_pow_ne_zero 0)
    (det_vandermonde_pow_ne_zero 1))
  exact_mod_cast Nat.cast_ne_zero.mpr (Nat.multinomial_pos _ _).ne'

/-! ### The main theorem -/

/-- **Theorem.**  The complete bipartite graph `K_{N,N}` has rank exactly
    `N (N - 1) / 2`, achieving the upper bound. -/
theorem rank_complete
    (E : Fin N → Fin N → Prop) [DecidableRel E]
    (hcomplete : IsCompleteBipartite E) :
    rank E = ((N * (N - 1) / 2 : ℕ) : ℕ∞) := by
  have htri : N * (N - 1) / 2 = ∑ i : Fin N, (i : ℕ) := by
    rw [Fin.sum_univ_eq_sum_range (fun j => j) N, Finset.sum_range_id]
  rw [rank, htri]
  apply le_antisymm
  · exact iInf₂_le (∑ i : Fin N, (i : ℕ))
      (show (∑ i : Fin N, (i : ℕ)) ∈ {m : ℕ | g E m ≠ 0} from g_triangle_ne_zero E hcomplete)
  · refine le_iInf₂ fun n hn => ?_
    have hle : ∑ i : Fin N, (i : ℕ) ≤ n := by
      by_contra h
      push_neg at h
      exact hn (g_eq_zero_of_lt E hcomplete h)
    exact_mod_cast hle

theorem rank_le_choose_two
    (E : Fin N → Fin N → Prop) [DecidableRel E]
    (hpm : ∃ π : Equiv.Perm (Fin N), IsPerfectMatching E π) :
    rank E ≤ ((N * (N - 1) / 2 : ℕ) : ℕ∞) := by
  sorry

end BipartiteRank
