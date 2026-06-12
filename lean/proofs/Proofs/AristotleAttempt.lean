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

/-! ## Auxiliary lemmas -/

lemma all_matchings_of_complete
    (E : Fin N → Fin N → Prop) (hcomplete : IsCompleteBipartite E) :
    ∀ π : Equiv.Perm (Fin N), IsPerfectMatching E π :=
  fun π i => hcomplete i (π i)

lemma g_complete_eq (E : Fin N → Fin N → Prop) [DecidableRel E]
    (hcomplete : IsCompleteBipartite E) (n : ℕ) :
    g E n = ∑ π : Equiv.Perm (Fin N), (Equiv.Perm.sign π : ℤ) • f π ^ n := by
  change ∑ π ∈ Finset.univ.filter (IsPerfectMatching E), _ = _
  rw [Finset.filter_true_of_mem fun x _ => all_matchings_of_complete E hcomplete x]

lemma signed_sum_eq_det (M : Matrix (Fin N) (Fin N) Poly) :
    ∑ π : Equiv.Perm (Fin N), (Equiv.Perm.sign π : ℤ) • ∏ i, M i (π i) = M.det := by
  rw [← Matrix.det_transpose, Matrix.det_apply']
  simp +decide [Matrix.transpose_apply, mul_comm]

instance : IsDomain Poly := inferInstance

lemma vandermonde_X_pow_ne_zero (v : Fin N → ℕ) (hv : Function.Injective v) :
    (Matrix.vandermonde (fun i => (X (1 : Fin 2) : Poly) ^ v i)).det ≠ 0 := by
  simp +decide [Matrix.det_vandermonde]
  simp +decide [Finset.prod_eq_zero_iff, sub_eq_zero]
  intro i j hij H
  replace H := congr_arg (MvPolynomial.eval (fun _ => 2)) H; norm_num at H
  exact hij.ne (hv H ▸ rfl)

lemma min_sum_distinct (α : Fin N → ℕ) (hinj : Function.Injective α) :
    N * (N - 1) / 2 ≤ ∑ i, α i := by
  have h_order : ∃ β : Fin N → ℕ, StrictMono β ∧ ∀ i, β i ∈ Finset.image α Finset.univ :=
    ⟨fun i ↦ Finset.orderEmbOfFin (Finset.image α Finset.univ)
      (by simp [Finset.card_image_of_injective _ hinj]) i,
     by simp +decide [StrictMono],
     fun i ↦ Finset.orderEmbOfFin_mem _ _ _⟩
  obtain ⟨β, hβ_mono, hβ_image⟩ := h_order
  have h_sum_eq : ∑ i, β i = ∑ i, α i := by
    have h_sum_eq : ∑ i ∈ Finset.univ.image β, i = ∑ i ∈ Finset.univ.image α, i := by
      rw [Finset.eq_of_subset_of_card_le (Finset.image_subset_iff.mpr fun i _ => hβ_image i)
        (by rw [Finset.card_image_of_injective _ hβ_mono.injective,
                Finset.card_image_of_injective _ hinj])]
    rwa [Finset.sum_image (by intros i _ j _ hij; exact hβ_mono.injective hij),
         Finset.sum_image (by intros i _ j _ hij; exact hinj hij)] at h_sum_eq
  have hβ_ge : ∀ i, β i ≥ i := by
    intro ⟨i, hi⟩; induction i with
    | zero => exact Nat.zero_le _
    | succ i ih => exact Nat.succ_le_of_lt (lt_of_le_of_lt (ih (Nat.lt_of_succ_lt hi))
        (hβ_mono (Nat.lt_succ_self _)))
  rw [← h_sum_eq, ← Finset.sum_range_id]
  simpa only [Finset.sum_range] using Finset.sum_le_sum fun i _ => hβ_ge i

lemma not_injective_of_sum_lt (α : Fin N → ℕ) (hsum : ∑ i, α i = n)
    (hlt : n < N * (N - 1) / 2) : ¬Function.Injective α :=
  fun hinj => not_lt_of_ge (min_sum_distinct α hinj) (hsum ▸ hlt)

lemma signed_perm_sum_zero_of_not_injective (α : Fin N → ℕ) (hα : ¬Function.Injective α) :
    ∑ π : Equiv.Perm (Fin N),
      (Equiv.Perm.sign π : ℤ) • (X (1 : Fin 2) : Poly) ^ ∑ i, α i * (π i : ℕ) = 0 := by
  convert signed_sum_eq_det (fun i j => (X 1 : Poly) ^ (α i * (j : ℕ))) using 1
  · simp +decide [Finset.prod_pow_eq_pow_sum]
  · obtain ⟨i, j, hij, h⟩ := Function.not_injective_iff.mp hα
    exact Eq.symm (Matrix.det_zero_of_row_eq h <| by aesop)

/-! ### Fiber cardinality -/

noncomputable def fiberCard {n : ℕ} (p : Fin n → Fin N) (j : Fin N) : ℕ :=
  (Finset.univ.filter (fun k => p k = j)).card

lemma fiberCard_sum (p : Fin n → Fin N) : ∑ j, fiberCard p j = n := by
  have : ∑ j : Fin N, (Finset.univ.filter (fun k => p k = j)).card =
      Finset.card (Finset.univ : Finset (Fin n)) := by
    simp +decide only [card_filter]
    rw [Finset.sum_comm]; simp +decide
  aesop

lemma sum_comp_eq_sum_fiberCard (p : Fin n → Fin N) (h : Fin N → ℕ) :
    ∑ k : Fin n, h (p k) = ∑ j : Fin N, fiberCard p j * h j := by
  simp +decide only [fiberCard, card_filter, Finset.sum_mul]
  rw [Finset.sum_comm]; aesop

/-! ### Vanishing -/

lemma signed_sum_for_tuple_eq_zero (p : Fin n → Fin N) (hn : n < N * (N - 1) / 2) :
    ∑ π : Equiv.Perm (Fin N),
      (Equiv.Perm.sign π : ℤ) •
        (∏ k : Fin n, (X (0 : Fin 2) : Poly) ^ (p k : ℕ) *
          (X (1 : Fin 2) : Poly) ^ (π (p k) : ℕ)) = 0 := by
  suffices h : ∑ π : Equiv.Perm (Fin N), (Equiv.Perm.sign π : ℤ) •
      ((∏ k, (X 0 : Poly) ^ (p k : ℕ)) *
       (X 1 : Poly) ^ (∑ j, fiberCard p j * (π j : ℕ))) = 0 by
    convert h using 3
    simp +decide [Finset.prod_mul_distrib, Finset.prod_pow_eq_pow_sum]
    rw [← Equiv.sum_comp (‹_› : Equiv.Perm (Fin N))]; simp +decide [fiberCard]
  convert congr_arg (fun x : Poly => (∏ k : Fin n, (X 0 : Poly) ^ (p k : ℕ)) * x)
    (signed_perm_sum_zero_of_not_injective (fiberCard p) ?_) using 1
  · simp +decide only [Finset.mul_sum, mul_smul_comm]
  · ring
  · exact not_injective_of_sum_lt _ (fiberCard_sum p) hn

lemma g_complete_eq_zero (E : Fin N → Fin N → Prop) [DecidableRel E]
    (hcomplete : IsCompleteBipartite E) {n : ℕ} (_hlt : n < N * (N - 1) / 2) :
    g E n = 0 := by
  have h : ∑ p : Fin n → Fin N, ∑ π : Equiv.Perm (Fin N),
      (Equiv.Perm.sign π : ℤ) •
        (∏ k : Fin n, (X (0 : Fin 2) : Poly) ^ (p k : ℕ) *
          (X (1 : Fin 2) : Poly) ^ (π (p k) : ℕ)) = 0 :=
    Finset.sum_eq_zero fun p _ => by simpa using signed_sum_for_tuple_eq_zero p hlt
  convert h using 2
  rw [g_complete_eq E hcomplete n]
  convert Finset.sum_comm using 2
  rename_i π _
  rw [show f π ^ n = (∑ k : Fin N, X 0 ^ (k : ℕ) * X 1 ^ (π k : ℕ)) ^ n by rfl,
      Fintype.sum_pow]
  exact Finset.smul_sum

/-! ### Non-vanishing -/

lemma signed_sum_eq_sign_mul_det (σ : Equiv.Perm (Fin N)) :
    ∑ π : Equiv.Perm (Fin N),
      (Equiv.Perm.sign π : ℤ) • (X (1 : Fin 2) : Poly) ^ ∑ j, (σ j : ℕ) * (π j : ℕ) =
    (Equiv.Perm.sign σ : ℤ) •
      (Matrix.vandermonde (fun i : Fin N => (X (1 : Fin 2) : Poly) ^ (i : ℕ))).det := by
  set M : Matrix (Fin N) (Fin N) Poly := fun j l => (X 1 : Poly) ^ ((σ j) * l.val)
  have h_det : ∑ π : Equiv.Perm (Fin N), (π.sign : ℤ) • ∏ j : Fin N, M j (π j) = M.det := by
    exact signed_sum_eq_det _
  convert h_det using 1
  · exact Finset.sum_congr rfl fun _ _ => by rw [← Finset.prod_pow_eq_pow_sum]
  · convert Matrix.det_permute σ (Matrix.vandermonde fun i : Fin N => (X 1 : Poly) ^ (i : ℕ))
      using 1
    · simp +decide [Matrix.det_apply']
      rw [Finset.mul_sum]
      refine Finset.sum_bij (fun π _ => σ⁻¹ * π) ?_ ?_ ?_ ?_ <;>
        simp +decide [mul_comm, mul_left_comm]
      exact fun b => ⟨σ * b, by simp +decide⟩
    · convert Matrix.det_permute σ (Matrix.vandermonde fun i : Fin N => (X 1 : Poly) ^ (i : ℕ))
        using 1
      exact congr_arg Matrix.det (by ext i j; simp +decide [M, pow_mul])

lemma x_exponent_of_perm (σ : Equiv.Perm (Fin N)) (p : Fin (N * (N - 1) / 2) → Fin N)
    (_hp : fiberCard p = fun j => (σ j : ℕ)) :
    ∑ k, (p k : ℕ) = ∑ j : Fin N, (j : ℕ) * (σ j : ℕ) := by
  sorry

lemma multinomial_coeff_pos :
    0 < (N * (N - 1) / 2).factorial / ∏ i : Fin N, (i : ℕ).factorial := by
  refine Nat.div_pos ?_ ?_
  · induction N <;>
      simp_all +decide [Fin.prod_univ_castSucc, Nat.mul_succ, Nat.mul_div_assoc]
    cases ‹ℕ› <;> simp_all +decide [Nat.mul_comm, Nat.mul_div_assoc]
    rename_i k hk
    refine le_trans (Nat.mul_le_mul_left _ hk) ?_
    refine' Nat.le_of_dvd ( Nat.factorial_pos _ ) ( Nat.factorial_mul_factorial_dvd_factorial_add _ _ |> dvd_trans <| Nat.factorial_dvd_factorial _ );
    lia
  · exact Finset.prod_pos fun _ _ => Nat.factorial_pos _

/-
The rearrangement inequality: `∑ j·σ(j) ≤ ∑ j²` for any permutation σ,
    with equality iff σ = id. Proof: `∑(j-σ(j))² = 2∑j² - 2∑j·σ(j) ≥ 0`.
-/
lemma sum_sq_sub_sum_mul_perm (σ : Equiv.Perm (Fin N)) :
    ∑ i : Fin N, ((i : ℤ) - (σ i : ℤ)) ^ 2 =
    2 * ∑ i : Fin N, (i : ℤ) ^ 2 - 2 * ∑ i : Fin N, (i : ℤ) * (σ i : ℤ) := by
  simp +decide only [sub_sq, mul_assoc, sum_add_distrib, sum_sub_distrib];
  rw [ Equiv.sum_comp σ fun i : Fin N => ( i : ℤ ) ^ 2 ] ; norm_num [ Finset.mul_sum _ _ _ ] ; ring;
  simpa only [ ← Finset.sum_mul _ _ _ ] using by ring;

lemma sum_mul_perm_le_sum_sq (σ : Equiv.Perm (Fin N)) :
    ∑ i : Fin N, (i : ℕ) * (σ i : ℕ) ≤ ∑ i : Fin N, (i : ℕ) * (i : ℕ) := by
  -- From sum_sq_sub_sum_mul_perm: ∑(i - σ(i))² = 2∑i² - 2∑i·σ(i).
  have h_sq_sub :
      ∑ i : Fin N, ((i : ℤ) - (σ i : ℤ)) ^ 2 =
      2 * ∑ i : Fin N, (i : ℤ) ^ 2 - 2 * ∑ i : Fin N, (i : ℤ) * (σ i : ℤ) := by
    convert sum_sq_sub_sum_mul_perm σ using 1;
  -- Since ∑(i-σ(i))² ≥ 0, we have 2∑i² - 2∑i·σ(i) ≥ 0, hence ∑i·σ(i) ≤ ∑i².
  have h_nonneg : 0 ≤ ∑ i : Fin N, ((i : ℤ) - (σ i : ℤ)) ^ 2 := by
    exact Finset.sum_nonneg fun _ _ => sq_nonneg _;
  norm_num [ ← sq ] at * ; linarith

lemma sum_mul_perm_eq_sum_sq_iff (σ : Equiv.Perm (Fin N)) :
    ∑ i : Fin N, (i : ℕ) * (σ i : ℕ) = ∑ i : Fin N, (i : ℕ) * (i : ℕ) ↔ σ = Equiv.refl _ := by
  constructor <;> intro h;
  · -- We show that each term in the sum $\sum (i - \sigma(i))^2$ is zero.
    have h_term_zero : ∀ i : Fin N, (i - σ i : ℤ) ^ 2 = 0 := by
      have h_term_zero : ∑ i : Fin N, ((i : ℤ) - (σ i : ℤ)) ^ 2 = 0 := by
        simp_all +decide [ sub_sq, Finset.sum_add_distrib ];
        simp_all +decide [ mul_assoc, sq ];
        linarith! [ Equiv.sum_comp σ fun i : Fin N => ( i : ℤ ) * i ];
      rw [ Finset.sum_eq_zero_iff_of_nonneg fun _ _ => sq_nonneg _ ] at h_term_zero
      aesop
    ext i; specialize h_term_zero i; simp_all +decide [ sub_eq_iff_eq_add ] ;
  · grind

/-
The sum `∑ j·σ(j)` over `Fin N` is strictly less than `∑ j²` when `σ ≠ id`.
-/
lemma sum_mul_perm_lt_of_ne (σ : Equiv.Perm (Fin N)) (hσ : σ ≠ Equiv.refl _) :
    ∑ i : Fin N, (i : ℕ) * (σ i : ℕ) < ∑ i : Fin N, (i : ℕ) * (i : ℕ) := by
  contrapose! hσ
  exact sum_mul_perm_eq_sum_sq_iff σ |>.1
    (le_antisymm (by simpa [ mul_comm ] using sum_mul_perm_le_sum_sq σ) hσ)

/-
Over `ℤ`, the signed sum `∑_π sign(π) * b^{∑ σ(j)*π(j)}` equals
    `sign(σ) * ∏_{i<j} (b^j - b^i)` for any base `b` and permutation `σ`.
-/
lemma signed_sum_eval (σ : Equiv.Perm (Fin N)) (b : ℤ) :
    ∑ π : Equiv.Perm (Fin N),
      (Equiv.Perm.sign π : ℤ) * b ^ ∑ j, (σ j : ℕ) * (π j : ℕ) =
    (Equiv.Perm.sign σ : ℤ) *
      ∏ i : Fin N, ∏ j ∈ Finset.Ioi i, (b ^ (j : ℕ) - b ^ (i : ℕ)) := by
  convert signed_sum_eq_sign_mul_det ( σ ) using 1;
  rw [ Matrix.det_vandermonde ];
  refine ⟨ fun h => ?_, fun h => ?_ ⟩;
  · convert signed_sum_eq_sign_mul_det σ using 1;
    rw [ Matrix.det_vandermonde ];
  · convert congr_arg ( MvPolynomial.eval ( fun _ => b ) ) h using 1 <;> norm_num

/-- The evaluation of `g E m` at `(2, 3)` factors as `C * V_2 * V_3` where
    `V_a = ∏_{i<j} (a^j - a^i)` and `C > 0`. -/
lemma eval_g_eq_prod (E : Fin N → Fin N → Prop) [DecidableRel E]
    (hcomplete : IsCompleteBipartite E) :
    ∃ (c : ℤ), 0 < c ∧
      MvPolynomial.eval (![2, 3] : Fin 2 → ℤ) (g E (N * (N - 1) / 2)) =
      c * (∏ i : Fin N, ∏ j ∈ Finset.Ioi i, ((2 : ℤ) ^ (j : ℕ) - (2 : ℤ) ^ (i : ℕ))) *
          (∏ i : Fin N, ∏ j ∈ Finset.Ioi i, ((3 : ℤ) ^ (j : ℕ) - (3 : ℤ) ^ (i : ℕ))) := by
  sorry

lemma g_complete_ne_zero (E : Fin N → Fin N → Prop) [DecidableRel E]
    (hcomplete : IsCompleteBipartite E) :
    g E (N * (N - 1) / 2) ≠ 0 := by
  obtain ⟨c, hc, heval⟩ := eval_g_eq_prod E hcomplete;
  contrapose! heval; simp_all +decide [ Finset.prod_eq_zero_iff, sub_eq_zero ] ;
  exact ⟨ hc.ne', fun i j hij => ne_of_gt hij ⟩

/-! ### Final theorem -/

lemma iInf_nat_eq_of_isLeast {S : Set ℕ} {m : ℕ} (hm : IsLeast S m) :
    ⨅ n ∈ S, (n : ℕ∞) = (m : ℕ∞) := by
  -- By definition of infimum, we know that for any $n \in S$, $n \geq m$.
  have h_inf_ge : ∀ n ∈ S, (n : ℝ) ≥ m := by
    exact fun n hn => Nat.cast_le.mpr ( hm.2 hn );
  refine le_antisymm ?_ ?_;
  · refine csInf_le ?_ ?_ <;> norm_num;
    exact ⟨ m, by simp +decide [ hm.1 ] ⟩;
  · refine le_ciInf fun n => ?_;
    by_cases hn : n ∈ S <;> aesop

/-- **Theorem.**  The complete bipartite graph `K_{N,N}` has rank exactly
    `N (N - 1) / 2`, achieving the upper bound. -/
theorem rank_complete
    (E : Fin N → Fin N → Prop) [DecidableRel E]
    (hcomplete : IsCompleteBipartite E) :
    rank E = ((N * (N - 1) / 2 : ℕ) : ℕ∞) := by
  refine le_antisymm ?_ ?_
  · refine ciInf_le_of_le ?_ ?_ ?_ <;> norm_num
    exacts [N * (N - 1) / 2, by simp +decide [g_complete_ne_zero E hcomplete]]
  · refine le_ciInf fun n => ?_
    by_cases hn : g E n = 0 <;> simp_all +decide
    exact le_of_not_gt fun h => hn <| g_complete_eq_zero E hcomplete h

end BipartiteRank
