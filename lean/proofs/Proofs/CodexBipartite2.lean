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

The complete-graph theorem is proved below.
-/
import Mathlib

open MvPolynomial BigOperators Finset

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

/-! ## Complete graphs and determinant cancellation -/

lemma all_matchings_of_complete
    (E : Fin N → Fin N → Prop) (hcomplete : IsCompleteBipartite E) :
    ∀ π : Equiv.Perm (Fin N), IsPerfectMatching E π :=
  fun π i => hcomplete i (π i)

lemma g_complete_eq (E : Fin N → Fin N → Prop) [DecidableRel E]
    (hcomplete : IsCompleteBipartite E) (n : ℕ) :
    g E n = ∑ π : Equiv.Perm (Fin N), (Equiv.Perm.sign π : ℤ) • f π ^ n := by
  show ∑ π ∈ Finset.univ.filter (IsPerfectMatching E), _ = _
  rw [Finset.filter_true_of_mem fun π _ => all_matchings_of_complete E hcomplete π]

lemma signed_sum_eq_det (M : Matrix (Fin N) (Fin N) Poly) :
    ∑ π : Equiv.Perm (Fin N), (Equiv.Perm.sign π : ℤ) • ∏ i, M i (π i) = M.det := by
  rw [← Matrix.det_transpose, Matrix.det_apply']
  simp +decide [Matrix.transpose_apply, mul_comm]

instance : IsDomain Poly := inferInstance

/-- The sum of `N` distinct natural numbers is at least `0 + ... + (N - 1)`. -/
lemma min_sum_distinct (a : Fin N → ℕ) (ha : Function.Injective a) :
    N * (N - 1) / 2 ≤ ∑ i, a i := by
  let s := Finset.image a Finset.univ
  have hs : s.card = N := by
    simp [s, Finset.card_image_of_injective _ ha]
  let b : Fin N → ℕ := s.orderEmbOfFin hs
  have hb_mem : ∀ i, b i ∈ s := fun i => Finset.orderEmbOfFin_mem _ _ _
  have hb_mono : StrictMono b := (s.orderEmbOfFin hs).strictMono
  have hsum : ∑ i, b i = ∑ i, a i := by
    calc
      ∑ i, b i = ∑ x ∈ Finset.image b Finset.univ, x := by
        rw [Finset.sum_image]
        exact fun _ _ _ _ h => hb_mono.injective h
      _ = ∑ x ∈ s, x := by
        rw [Finset.image_orderEmbOfFin_univ]
      _ = ∑ i, a i := by
        rw [Finset.sum_image]
        exact fun _ _ _ _ h => ha h
  have hb_ge : ∀ i : Fin N, (i : ℕ) ≤ b i := by
    intro ⟨i, hi⟩
    induction' i with i ih
    · exact Nat.zero_le _
    · exact Nat.succ_le_of_lt
        (lt_of_le_of_lt (ih (Nat.lt_of_succ_lt hi))
          (hb_mono (Nat.lt_succ_self _)))
  rw [← hsum, ← Finset.sum_range_id]
  simpa only [Finset.sum_range] using
    Finset.sum_le_sum fun i _ => hb_ge i

lemma signed_perm_sum_zero_of_not_injective
    (a : Fin N → ℕ) (ha : ¬Function.Injective a) :
    ∑ π : Equiv.Perm (Fin N),
      (Equiv.Perm.sign π : ℤ) •
        (X (1 : Fin 2) : Poly) ^ ∑ i, a i * (π i : ℕ) = 0 := by
  convert signed_sum_eq_det
    (fun i j => (X (1 : Fin 2) : Poly) ^ (a i * (j : ℕ))) using 1
  · simp +decide [Finset.prod_pow_eq_pow_sum]
  · obtain ⟨i, j, hij, h⟩ := Function.not_injective_iff.mp ha
    exact Eq.symm (Matrix.det_zero_of_row_eq h <| by aesop)

noncomputable def fiberCard {n : ℕ} (p : Fin n → Fin N) (j : Fin N) : ℕ :=
  (Finset.univ.filter (fun k => p k = j)).card

lemma fiberCard_sum {n : ℕ} (p : Fin n → Fin N) : ∑ j, fiberCard p j = n := by
  have h :
      ∑ j : Fin N, (Finset.univ.filter (fun k => p k = j)).card =
        Finset.card (Finset.univ : Finset (Fin n)) := by
    simp +decide only [card_filter]
    rw [Finset.sum_comm]
    simp +decide
  simpa [fiberCard] using h

lemma sum_comp_eq_sum_fiberCard {n : ℕ} (p : Fin n → Fin N) (h : Fin N → ℕ) :
    ∑ k : Fin n, h (p k) = ∑ j : Fin N, fiberCard p j * h j := by
  simp +decide only [fiberCard, card_filter, Finset.sum_mul]
  rw [Finset.sum_comm]
  aesop

lemma signed_sum_for_tuple_eq_zero
    {n : ℕ} (p : Fin n → Fin N) (hn : n < N * (N - 1) / 2) :
    ∑ π : Equiv.Perm (Fin N),
      (Equiv.Perm.sign π : ℤ) •
        (∏ k : Fin n, (X (0 : Fin 2) : Poly) ^ (p k : ℕ) *
          (X (1 : Fin 2) : Poly) ^ (π (p k) : ℕ)) = 0 := by
  suffices h :
      ∑ π : Equiv.Perm (Fin N), (Equiv.Perm.sign π : ℤ) •
        ((∏ k, (X (0 : Fin 2) : Poly) ^ (p k : ℕ)) *
          (X (1 : Fin 2) : Poly) ^
            (∑ j, fiberCard p j * (π j : ℕ))) = 0 by
    convert h using 3
    simp +decide [Finset.prod_mul_distrib, Finset.prod_pow_eq_pow_sum,
      sum_comp_eq_sum_fiberCard]
    rw [← Equiv.sum_comp (‹_› : Equiv.Perm (Fin N))]
    simp +decide [fiberCard]
  have hnoninj : ¬Function.Injective (fiberCard p) := by
    intro hinj
    have hmin := min_sum_distinct (fiberCard p) hinj
    rw [fiberCard_sum] at hmin
    omega
  convert congr_arg
    (fun q : Poly => (∏ k : Fin n, (X (0 : Fin 2) : Poly) ^ (p k : ℕ)) * q)
    (signed_perm_sum_zero_of_not_injective (fiberCard p) hnoninj) using 1
  · simp +decide only [Finset.mul_sum, mul_smul_comm]
  · ring

lemma g_complete_eq_zero (E : Fin N → Fin N → Prop) [DecidableRel E]
    (hcomplete : IsCompleteBipartite E) {n : ℕ}
    (hn : n < N * (N - 1) / 2) :
    g E n = 0 := by
  have h :
      ∑ p : Fin n → Fin N, ∑ π : Equiv.Perm (Fin N),
        (Equiv.Perm.sign π : ℤ) •
          (∏ k : Fin n, (X (0 : Fin 2) : Poly) ^ (p k : ℕ) *
            (X (1 : Fin 2) : Poly) ^ (π (p k) : ℕ)) = 0 :=
    Finset.sum_eq_zero fun p _ => by
      simpa using signed_sum_for_tuple_eq_zero p hn
  convert h using 2
  rw [g_complete_eq E hcomplete n]
  convert Finset.sum_comm using 2
  rename_i π _
  rw [show f π ^ n =
    (∑ k : Fin N,
      X (0 : Fin 2) ^ (k : ℕ) * X (1 : Fin 2) ^ (π k : ℕ)) ^ n by rfl,
    Fintype.sum_pow]
  exact Finset.smul_sum

/-! ## The critical coefficient -/

/-- At the minimum possible sum, an injective count vector is a permutation of
`0, ..., N - 1`. -/
lemma injective_eq_perm_of_sum_eq
    (a : Fin N → ℕ) (ha : Function.Injective a)
    (hsum : ∑ i, a i = N * (N - 1) / 2) :
    ∃ σ : Equiv.Perm (Fin N), ∀ i, a i = (σ i : ℕ) := by
  let s := Finset.image a Finset.univ
  have hs : s.card = N := by
    simp [s, Finset.card_image_of_injective _ ha]
  let b : Fin N → ℕ := s.orderEmbOfFin hs
  have hb_mono : StrictMono b := (s.orderEmbOfFin hs).strictMono
  have hb_sum : ∑ i, b i = ∑ i, a i := by
    calc
      ∑ i, b i = ∑ x ∈ Finset.image b Finset.univ, x := by
        rw [Finset.sum_image]
        exact fun _ _ _ _ h => hb_mono.injective h
      _ = ∑ x ∈ s, x := by
        rw [Finset.image_orderEmbOfFin_univ]
      _ = ∑ i, a i := by
        rw [Finset.sum_image]
        exact fun _ _ _ _ h => ha h
  have hb_ge : ∀ i : Fin N, (i : ℕ) ≤ b i := by
    intro ⟨i, hi⟩
    induction' i with i ih
    · exact Nat.zero_le _
    · exact Nat.succ_le_of_lt
        (lt_of_le_of_lt (ih (Nat.lt_of_succ_lt hi))
          (hb_mono (Nat.lt_succ_self _)))
  have hfin_sum : ∑ i : Fin N, (i : ℕ) = N * (N - 1) / 2 := by
    simpa only [Finset.sum_range] using Finset.sum_range_id N
  have hsum_eq : ∑ i : Fin N, (i : ℕ) = ∑ i, b i := by
    rw [hfin_sum, hb_sum, hsum]
  have hb_eq : ∀ i : Fin N, (i : ℕ) = b i := by
    intro i
    exact (Finset.sum_eq_sum_iff_of_le
      (s := (Finset.univ : Finset (Fin N)))
      (fun j _ => hb_ge j)).mp hsum_eq i (Finset.mem_univ i)
  have ha_lt : ∀ i, a i < N := by
    intro i
    have hai : a i ∈ s := by simp [s]
    rw [← Finset.image_orderEmbOfFin_univ s hs] at hai
    obtain ⟨j, _, hj⟩ := Finset.mem_image.mp hai
    rw [← hj]
    change b j < N
    rw [← hb_eq j]
    exact j.isLt
  let q : Fin N → Fin N := fun i => ⟨a i, ha_lt i⟩
  have hq_inj : Function.Injective q := by
    intro i j hij
    apply ha
    exact Fin.ext_iff.mp hij
  have hq_bij : Function.Bijective q :=
    (Fintype.bijective_iff_injective_and_card q).2 ⟨hq_inj, rfl⟩
  let σ : Equiv.Perm (Fin N) := Equiv.ofBijective q hq_bij
  exact ⟨σ, fun _ => rfl⟩

lemma sum_sq_sub_sum_mul_perm (σ : Equiv.Perm (Fin N)) :
    ∑ i : Fin N, ((i : ℤ) - (σ i : ℤ)) ^ 2 =
      2 * ∑ i : Fin N, (i : ℤ) ^ 2 -
        2 * ∑ i : Fin N, (i : ℤ) * (σ i : ℤ) := by
  simp +decide only [sub_sq, mul_assoc, sum_add_distrib, sum_sub_distrib]
  rw [Equiv.sum_comp σ fun i : Fin N => (i : ℤ) ^ 2]
  norm_num [Finset.mul_sum]
  ring
  simpa only [← Finset.sum_mul] using (by ring)

lemma sum_mul_perm_le_sum_sq (σ : Equiv.Perm (Fin N)) :
    ∑ i : Fin N, (i : ℕ) * (σ i : ℕ) ≤
      ∑ i : Fin N, (i : ℕ) * (i : ℕ) := by
  have hmono :
      Monovary (fun i : Fin N => (i : ℕ)) (fun i : Fin N => (i : ℕ)) := by
    intro _ _ hij
    exact Nat.le_of_lt hij
  exact Monovary.sum_mul_comp_perm_le_sum_mul hmono

lemma sum_mul_perm_eq_sum_sq_iff (σ : Equiv.Perm (Fin N)) :
    (∑ i : Fin N, (i : ℕ) * (σ i : ℕ)) =
        ∑ i : Fin N, (i : ℕ) * (i : ℕ) ↔
      σ = Equiv.refl _ := by
  constructor
  · intro h
    have hZ :
        (∑ i : Fin N, (i : ℤ) * (σ i : ℤ)) =
          ∑ i : Fin N, (i : ℤ) * (i : ℤ) := by
      exact_mod_cast h
    have hzero : ∑ i : Fin N, ((i : ℤ) - (σ i : ℤ)) ^ 2 = 0 := by
      rw [sum_sq_sub_sum_mul_perm]
      rw [hZ]
      ring
    have hterm :
        ∀ i : Fin N, ((i : ℤ) - (σ i : ℤ)) ^ 2 = 0 := by
      rw [Finset.sum_eq_zero_iff_of_nonneg
        (fun _ _ => sq_nonneg _)] at hzero
      simpa using hzero
    apply Equiv.ext
    intro i
    apply Fin.ext
    have hi := hterm i
    have hieq : (i : ℤ) = (σ i : ℤ) := by nlinarith
    exact_mod_cast hieq.symm
  · rintro rfl
    rfl

def xExponent (a : Fin N → ℕ) : ℕ :=
  ∑ i : Fin N, (i : ℕ) * a i

def yExponent (a : Fin N → ℕ) (π : Equiv.Perm (Fin N)) : ℕ :=
  ∑ i : Fin N, a i * (π i : ℕ)

def squareExponent (N : ℕ) : ℕ :=
  ∑ i : Fin N, (i : ℕ) * (i : ℕ)

noncomputable def targetMonomial (N : ℕ) : Fin 2 →₀ ℕ :=
  Finsupp.single (0 : Fin 2) (squareExponent N) +
    Finsupp.single (1 : Fin 2) (squareExponent N)

lemma two_single_eq_iff (a b c d : ℕ) :
    Finsupp.single (0 : Fin 2) a + Finsupp.single (1 : Fin 2) b =
        Finsupp.single (0 : Fin 2) c + Finsupp.single (1 : Fin 2) d ↔
      a = c ∧ b = d := by
  constructor
  · intro h
    constructor
    · have h0 := DFunLike.congr_fun h (0 : Fin 2)
      simpa using h0
    · have h1 := DFunLike.congr_fun h (1 : Fin 2)
      simpa using h1
  · rintro ⟨rfl, rfl⟩
    rfl

lemma matching_term_eq
    (π : Equiv.Perm (Fin N)) (a : Fin N → ℕ) :
    ∏ i : Fin N,
        ((X (0 : Fin 2) : Poly) ^ (i : ℕ) *
          X (1 : Fin 2) ^ (π i : ℕ)) ^ a i =
      X (0 : Fin 2) ^ xExponent a *
        X (1 : Fin 2) ^ yExponent a π := by
  simp_rw [mul_pow, ← pow_mul]
  rw [Finset.prod_mul_distrib]
  simp [xExponent, yExponent, Finset.prod_pow_eq_pow_sum, mul_comm]

lemma coeff_two_powers (a b : ℕ) :
    MvPolynomial.coeff (targetMonomial N)
        ((X (0 : Fin 2) : Poly) ^ a * X (1 : Fin 2) ^ b) =
      if a = squareExponent N ∧ b = squareExponent N then 1 else 0 := by
  simp [targetMonomial, MvPolynomial.X_pow_eq_monomial,
    MvPolynomial.monomial_mul, MvPolynomial.coeff_monomial,
    two_single_eq_iff]

lemma identity_count_mem_piAntidiag :
    (fun i : Fin N => (i : ℕ)) ∈
      (Finset.univ : Finset (Fin N)).piAntidiag (N * (N - 1) / 2) := by
  simp only [Finset.mem_piAntidiag]
  constructor
  · simpa only [Finset.sum_range] using Finset.sum_range_id N
  · simp

lemma signed_y_indicator_zero
    (a : Fin N → ℕ) (ha : ¬Function.Injective a) (d : ℕ) :
    ∑ π : Equiv.Perm (Fin N), (Equiv.Perm.sign π : ℤ) *
        (if yExponent a π = d then 1 else 0) = 0 := by
  have h := congr_arg
    (MvPolynomial.coeff (Finsupp.single (1 : Fin 2) d))
    (signed_perm_sum_zero_of_not_injective a ha)
  have hterm (π : Equiv.Perm (Fin N)) :
      MvPolynomial.coeff (Finsupp.single (1 : Fin 2) d)
          ((Equiv.Perm.sign π : ℤ) •
            (X (1 : Fin 2) : Poly) ^ ∑ i, a i * (π i : ℕ)) =
        (Equiv.Perm.sign π : ℤ) *
          (if ∑ i, a i * (π i : ℕ) = d then 1 else 0) := by
    rw [MvPolynomial.coeff_smul]
    have hsingle :
        Finsupp.single (1 : Fin 2) d =
            Finsupp.single (1 : Fin 2) (∑ i, a i * (π i : ℕ)) ↔
          d = ∑ i, a i * (π i : ℕ) := by
      constructor
      · intro hs
        have hs1 := DFunLike.congr_fun hs (1 : Fin 2)
        simpa using hs1
      · intro hs
        rw [hs]
    simp [MvPolynomial.coeff_X_pow, hsingle, eq_comm]
  rw [MvPolynomial.coeff_sum] at h
  simp_rw [hterm] at h
  simpa [yExponent] using h

lemma critical_signed_coeff
    (a : Fin N → ℕ)
    (ha_mem : a ∈
      (Finset.univ : Finset (Fin N)).piAntidiag (N * (N - 1) / 2)) :
    ∑ π : Equiv.Perm (Fin N), (Equiv.Perm.sign π : ℤ) *
        MvPolynomial.coeff (targetMonomial N)
          (∏ i : Fin N,
            ((X (0 : Fin 2) : Poly) ^ (i : ℕ) *
              X (1 : Fin 2) ^ (π i : ℕ)) ^ a i) =
      if a = (fun i : Fin N => (i : ℕ)) then 1 else 0 := by
  simp_rw [matching_term_eq, coeff_two_powers]
  by_cases ha : Function.Injective a
  · have hsum : ∑ i, a i = N * (N - 1) / 2 :=
      (Finset.mem_piAntidiag.mp ha_mem).1
    obtain ⟨σ, hσ⟩ := injective_eq_perm_of_sum_eq a ha hsum
    by_cases haid : a = (fun i : Fin N => (i : ℕ))
    · subst a
      simp [xExponent, yExponent, squareExponent,
        sum_mul_perm_eq_sum_sq_iff]
    · have hσne : σ ≠ Equiv.refl _ := by
        intro h
        apply haid
        funext i
        simpa [h] using hσ i
      have hxne : xExponent a ≠ squareExponent N := by
        intro hx
        apply hσne
        apply (sum_mul_perm_eq_sum_sq_iff σ).mp
        simpa [xExponent, squareExponent, hσ] using hx
      simp [hxne, haid]
  · have haid : a ≠ (fun i : Fin N => (i : ℕ)) := by
      intro h
      apply ha
      rw [h]
      exact Fin.val_injective
    rw [if_neg haid]
    by_cases hx : xExponent a = squareExponent N
    · simp only [hx, true_and]
      exact signed_y_indicator_zero a ha (squareExponent N)
    · simp [hx]

lemma f_pow_expansion (π : Equiv.Perm (Fin N)) (n : ℕ) :
    f π ^ n =
      ∑ a ∈ (Finset.univ : Finset (Fin N)).piAntidiag n,
        (Nat.multinomial Finset.univ a : ℤ) •
          ∏ i : Fin N,
            ((X (0 : Fin 2) : Poly) ^ (i : ℕ) *
              X (1 : Fin 2) ^ (π i : ℕ)) ^ a i := by
  unfold f
  rw [Finset.sum_pow_eq_sum_piAntidiag]
  simp [MvPolynomial.smul_eq_C_mul]

lemma g_critical_expansion
    (E : Fin N → Fin N → Prop) [DecidableRel E]
    (hcomplete : IsCompleteBipartite E) :
    g E (N * (N - 1) / 2) =
      ∑ a ∈
          (Finset.univ : Finset (Fin N)).piAntidiag (N * (N - 1) / 2),
        (Nat.multinomial Finset.univ a : ℤ) •
          ∑ π : Equiv.Perm (Fin N), (Equiv.Perm.sign π : ℤ) •
            ∏ i : Fin N,
              ((X (0 : Fin 2) : Poly) ^ (i : ℕ) *
                X (1 : Fin 2) ^ (π i : ℕ)) ^ a i := by
  rw [g_complete_eq E hcomplete]
  simp_rw [f_pow_expansion, Finset.smul_sum, smul_smul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro a _
  apply Finset.sum_congr rfl
  intro π _
  congr 1
  ring

lemma coeff_g_critical
    (E : Fin N → Fin N → Prop) [DecidableRel E]
    (hcomplete : IsCompleteBipartite E) :
    MvPolynomial.coeff (targetMonomial N)
        (g E (N * (N - 1) / 2)) =
      (Nat.multinomial Finset.univ (fun i : Fin N => (i : ℕ)) : ℤ) := by
  rw [g_critical_expansion E hcomplete]
  rw [MvPolynomial.coeff_sum]
  calc
    _ = ∑ a ∈
          (Finset.univ : Finset (Fin N)).piAntidiag (N * (N - 1) / 2),
        (Nat.multinomial Finset.univ a : ℤ) *
          ∑ π : Equiv.Perm (Fin N), (Equiv.Perm.sign π : ℤ) *
            MvPolynomial.coeff (targetMonomial N)
              (∏ i : Fin N,
                ((X (0 : Fin 2) : Poly) ^ (i : ℕ) *
                  X (1 : Fin 2) ^ (π i : ℕ)) ^ a i) := by
      apply Finset.sum_congr rfl
      intro a _
      rw [MvPolynomial.coeff_smul, MvPolynomial.coeff_sum]
      congr 1
    _ = ∑ a ∈
          (Finset.univ : Finset (Fin N)).piAntidiag (N * (N - 1) / 2),
        (Nat.multinomial Finset.univ a : ℤ) *
          (if a = (fun i : Fin N => (i : ℕ)) then 1 else 0) := by
      apply Finset.sum_congr rfl
      intro a ha
      rw [critical_signed_coeff a ha]
    _ = (Nat.multinomial Finset.univ (fun i : Fin N => (i : ℕ)) : ℤ) := by
      rw [Finset.sum_eq_single (fun i : Fin N => (i : ℕ))]
      · simp
      · intro a ha hne
        simp [hne]
      · intro hnot
        exact (hnot identity_count_mem_piAntidiag).elim

lemma g_complete_ne_zero
    (E : Fin N → Fin N → Prop) [DecidableRel E]
    (hcomplete : IsCompleteBipartite E) :
    g E (N * (N - 1) / 2) ≠ 0 := by
  intro hzero
  have hcoeff := coeff_g_critical E hcomplete
  rw [hzero] at hcoeff
  simp only [MvPolynomial.coeff_zero] at hcoeff
  have hpos :
      0 < Nat.multinomial (Finset.univ : Finset (Fin N))
        (fun i : Fin N => (i : ℕ)) :=
    Nat.multinomial_pos _ _
  have hne :
      (Nat.multinomial (Finset.univ : Finset (Fin N))
        (fun i : Fin N => (i : ℕ)) : ℤ) ≠ 0 := by
    exact_mod_cast (Nat.ne_of_gt hpos)
  exact hne hcoeff.symm

/-- **Theorem.**  The complete bipartite graph `K_{N,N}` has rank exactly
    `N (N - 1) / 2`, achieving the upper bound. -/
theorem rank_complete
    (E : Fin N → Fin N → Prop) [DecidableRel E]
    (hcomplete : IsCompleteBipartite E) :
    rank E = ((N * (N - 1) / 2 : ℕ) : ℕ∞) := by
  apply le_antisymm
  · unfold rank
    refine iInf_le_of_le (N * (N - 1) / 2) ?_
    refine iInf_le_of_le (g_complete_ne_zero E hcomplete) ?_
    exact le_rfl
  · unfold rank
    refine le_iInf fun n => ?_
    refine le_iInf fun hn => ?_
    have hnat : N * (N - 1) / 2 ≤ n := by
      apply Nat.le_of_not_gt
      intro hlt
      exact hn (g_complete_eq_zero E hcomplete hlt)
    exact_mod_cast hnat

/-- **Theorem.**  Every bipartite graph `G = (U, V, E)` with `|U| = |V| = N`
    that admits at least one perfect matching satisfies
    `rank G ≤ N (N - 1) / 2`   ( = `Nat.choose N 2`). -/
theorem rank_le_choose_two
    (E : Fin N → Fin N → Prop) [DecidableRel E]
    (hpm : ∃ π : Equiv.Perm (Fin N), IsPerfectMatching E π) :
    rank E ≤ ((N * (N - 1) / 2 : ℕ) : ℕ∞) := by
  sorry
end BipartiteRank
