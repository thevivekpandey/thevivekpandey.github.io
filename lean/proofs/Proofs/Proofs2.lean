import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Tactic

/--
Proof that there is no rational number whose square equals 2.
This is the classical proof by contradiction using coprimality.
-/

-- Helper lemma: if m² = 2n² with m, n coprime naturals, derive a contradiction
lemma no_coprime_sq_eq_two {m n : ℕ} (coprime_mn : m.Coprime n) : m ^ 2 ≠ 2 * n ^ 2 := by
  intro h

  -- Since m² = 2n², we have 2 ∣ m²
  have : 2 ∣ m ^ 2 := ⟨n ^ 2, h⟩
  -- Since 2 is prime, 2 ∣ m
  have h_m : 2 ∣ m := Nat.Prime.dvd_of_dvd_pow Nat.prime_two this
  -- So m = 2k
  obtain ⟨k, rfl⟩ := h_m
  -- Then (2k)² = 2n², so 4k² = 2n², hence n² = 2k²
  have h_key : 4 * k ^ 2 = 2 * n ^ 2 := by nlinarith [h]
  have : n ^ 2 = 2 * k ^ 2 := by omega
  -- So 2 ∣ n²
  have : 2 ∣ n ^ 2 := ⟨k ^ 2, this⟩
  -- Since 2 is prime, 2 ∣ n
  have h_n : 2 ∣ n := Nat.Prime.dvd_of_dvd_pow Nat.prime_two this
  -- But then 2 ∣ gcd(m, n) = 1
  have : 2 ∣ (2 * k).gcd n := Nat.dvd_gcd (by simp) h_n
  rw [coprime_mn] at this
  norm_num at this

-- Main theorem: no rational squares to 2
theorem no_rat_sq_eq_two : ¬ ∃ (q : ℚ), q * q = 2 := by
  intro ⟨q, hq⟩

  -- Derive m² = 2n² where m, n are coprime
  have h_eq : q.num.natAbs ^ 2 = 2 * q.den ^ 2 := by
    have h1 : q = q.num / q.den := q.num_div_den.symm
    rw [h1] at hq
    field_simp at hq
    have h2 : (q.num : ℚ) ^ 2 = 2 * (q.den : ℚ) ^ 2 := by
      convert hq using 1 <;> ring
    have h3 : ((q.num ^ 2 : ℤ) : ℚ) = ((2 * q.den ^ 2 : ℤ) : ℚ) := by
      push_cast
      convert h2 using 1 <;> ring
    have h4 : q.num ^ 2 = 2 * (q.den : ℤ) ^ 2 := Int.cast_injective h3
    have h5 : q.num.natAbs ^ 2 = (2 * (q.den : ℤ) ^ 2).natAbs := by
      rw [← Int.natAbs_pow, ← h4]
    rw [Int.natAbs_mul, Int.natAbs_pow] at h5
    simp at h5
    exact h5

  -- Apply lemma
  exact no_coprime_sq_eq_two q.reduced h_eq

#check no_rat_sq_eq_two
