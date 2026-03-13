import Mathlib

set_option linter.style.commandStart false

#check Nat

theorem infinitely_many_primes (n : Nat) : ∃ p, p ≥ n ∧ Nat.Prime p := by
  let m := Nat.factorial n + 1
  have hfact : Nat.factorial n ≥ 1 := Nat.factorial_pos n
  have hm : m ≥ 2 := by omega
  let p := m.minFac
  have hp : Nat.Prime p := Nat.minFac_prime (by omega)
  have hpm : p ∣ m := Nat.minFac_dvd m
  have hpn : p > n := by
    by_contra h
    push_neg at h
    have hpf : p ∣ Nat.factorial n := by
      apply (Nat.Prime.dvd_factorial hp).mpr
      omega
    have hpdvd1 : p ∣ 1 := by
      have h1 : p ∣ m - Nat.factorial n := Nat.dvd_sub hpm hpf
      have h2 : m - Nat.factorial n = 1 := by
        change Nat.factorial n + 1 - Nat.factorial n = 1
        omega
      rwa [h2] at h1
    have hple1 : p ≤ 1 := Nat.le_of_dvd (by omega) hpdvd1
    have hpge2 : p ≥ 2 := hp.two_le
    omega
  exact ⟨p, by omega, hp⟩
#print axioms infinitely_many_primes

theorem hom_maps_identity {G H : Type} [Group G] [Group H]
   (f : G →* H): f 1 = 1 := f.map_one

theorem hom_maps_identity1 {G H : Type} [Group G] [Group H]
    (f : G →* H) : f 1 = 1 := by
  have h1 : f 1 * f 1 = f (1 * 1) := (f.map_mul 1 1).symm
  have h2 : f (1 * 1) = f 1 := by rw [one_mul]
  have h3 : f 1 * f 1 = f 1 := by rw [h1, h2]
  have h4 : f 1 * f 1 = f 1 * 1 := by rw [h3, mul_one]
  exact mul_left_cancel h4

#check MonoidHom.map_mul

theorem hom_maps_identity2 {G H : Type} [Group G] [Group H]
    (f : G →* H) : f 1 = 1 := by
  have h1 : f 1 * f 1 = f 1 * 1 := by
    calc f 1 * f 1
        = f (1 * 1) := by rw [f.map_mul]
      _ = f 1       := by rw [one_mul]
      _ = f 1 * 1   := by rw [mul_one]
  exact mul_left_cancel h1

theorem identity_unique {G : Type} [Group G]
    (e : G) (he : ∀ a : G, e * a = a) : e = 1 := by
   have h1 : e * e = e := he e
   have h2 : e * e = e * 1 := by rw[h1, mul_one]
   exact mul_left_cancel h2

theorem inverse_unique {G : Type} [Group G]
   (e1 e2 e : G)
   (h1: e * e1 = 1)
   (h2: e2 * e = 1) :
   e1 = e2 := by calc
   e1 = 1 * e1        := by rw [one_mul]
    _ = (e2 * e) * e1 := by rw [h2]
    _ = e2 * (e * e1) := by rw [mul_assoc]
    _ = e2 * 1        := by rw [h1]
    _ = e2            := by rw[mul_one]

#check inv_mul_cancel

theorem inv_inv1 {G : Type} [Group G]
    (a : G) : a⁻¹⁻¹ = a := by
  have h1 : a⁻¹ * a⁻¹⁻¹ = 1 := mul_inv_cancel a⁻¹
  have h2 : a * a⁻¹ = 1 := mul_inv_cancel a
  exact inverse_unique a⁻¹⁻¹ a a⁻¹ h1 h2

theorem inv_prod {G: Type} [Group G]
   (a b: G) : (a * b)⁻¹ = b⁻¹ * a⁻¹ := by

  have h1: (a * b) * (a * b)⁻¹ = 1 := mul_inv_cancel (a * b)

  have h2: (b⁻¹ * a⁻¹) * (a * b) = 1 := by calc
      (b⁻¹ * a⁻¹) * (a * b)
          = b⁻¹ * (a⁻¹ * (a * b)) := by rw[mul_assoc]
        _ = b⁻¹ * ((a⁻¹ * a) * b) := by rw[mul_assoc]
        _ = 1 := by simp

  exact inverse_unique (a * b)⁻¹ (b⁻¹ * a⁻¹) (a * b) h1 h2

#check Nat.Prime

theorem inter_subgroup {G : Type} [Group G] (H K : Subgroup G):
  ∀ a b, a ∈ H ⊓ K → b ∈ H ⊓ K → a * b ∈ H ⊓ K := by
   intro a b ha hb
   obtain ⟨ah, ak⟩ := ha
   obtain ⟨bh, bk⟩ := hb
   have h1 : a * b ∈ H := Subgroup.mul_mem H ah bh
   have h2 : a * b ∈ K := Subgroup.mul_mem K ak bk
   exact Subgroup.mem_inf.mpr ⟨h1, h2⟩
