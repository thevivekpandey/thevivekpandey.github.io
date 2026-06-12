import Mathlib

open Finset MvPolynomial

variable {N : ℕ}

lemma finset_sum_ge_triangular (s : Finset ℕ) :
    ∑ x ∈ s, x ≥ s.card * (s.card - 1) / 2 := by
  induction' h_card : s.card with n ih generalizing s
  · have : s = ∅ := Finset.card_eq_zero.mp h_card
    rw [this]
    simp
  · have h_nonempty : s.Nonempty := by
      rw [← Finset.card_pos, h_card]
      exact n.succ_pos
    let M := s.max' h_nonempty
    have hM : M ∈ s := Finset.max'_mem s h_nonempty
    let s' := s.erase M
    have hcard_s' : s'.card = n := by
      rw [Finset.card_erase_of_mem hM, h_card]
      rfl
    have h_sum_s : ∑ x ∈ s, x = M + ∑ x ∈ s', x := by
      rw [← Finset.insert_erase hM, Finset.sum_insert (Finset.notMem_erase M s)]
    rw [h_sum_s]
    have h_M_ge : M ≥ n := by
      by_contra hlt
      push_neg at hlt
      have h_sub : s ⊆ Finset.range n := by
        intro x hx
        rw [Finset.mem_range]
        have h_le : x ≤ M := Finset.le_max' s hx
        linarith
      have h_card_le : s.card ≤ (Finset.range n).card := Finset.card_le_card h_sub
      rw [h_card, Finset.card_range] at h_card_le
      linarith
    have ih_s' := ih s' hcard_s'
    have h_M_sum : 2 * (M + ∑ x ∈ s', x) ≥ 2 * (n + n * (n - 1) / 2) := by omega
    have h_step : 2 * (n + n * (n - 1) / 2) = n * (n + 1) := by
      rcases Nat.even_or_odd n with h | h
      · obtain ⟨k, hk⟩ := h
        rw [hk]
        omega
      · obtain ⟨k, hk⟩ := h
        rw [hk]
        omega
    have h_rw : (n + 1) * ((n + 1) - 1) = n * (n + 1) := by omega
    have h_rw2 : (n + 1) * ((n + 1) - 1) / 2 * 2 = (n + 1) * ((n + 1) - 1) := by
      have : Even ((n + 1) * ((n + 1) - 1)) := by
        have : (n + 1) * ((n + 1) - 1) = n * (n + 1) := by omega
        rw [this]
        exact Nat.even_mul_succ_self n
      exact Nat.div_mul_cancel this.two_dvd
    have h_final : 2 * (M + ∑ x ∈ s', x) ≥ 2 * ((n + 1) * ((n + 1) - 1) / 2) := by
      rw [h_rw2, h_rw, ← h_step]
      exact h_M_sum
    omega
