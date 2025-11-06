import Proofs.Basic
import Proofs.Proofs2
import Mathlib.Tactic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Int.Basic

--import Mathlib.Data.Rat.Basic

example (a b : ℕ) : a + b = b + a := by ring

example (x y : Nat) : (x + y)^2 = x^2 + 2*x*y + y^2 := by ring

example (a b : Nat) (h : a ≤ b) : a ≤ b + 1 :=
   calc
      a ≤ b := h
      _ ≤ b + 1 := Nat.le_succ b

example (a b : Nat) (h : a ≤ b) : a ≤ b + 1 := by omega

example (a b c : Nat) (h1 : a ≤ b) (h2 : b ≤ c) : a ≤ c := by omega

example (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) : a ≤ c := by
  calc
  a ≤ b := h1
  _ ≤ c := h2

example (a b c : Nat) (h : a + b ≤ c) : a ≤ c := by
  calc
  a ≤ a + b := Nat.le_add_right a b
  _ ≤ c := h

example (n : Nat) : n = 0 ∨ n ≥ 1 := by
   cases n with
   | zero =>
     left
     rfl
   | succ k =>
     right
     omega

example (n : Nat) : n = 0 ∨ n > 0 := by
  cases n with
  | zero =>
    left
    rfl
  | succ k =>
    right
    simp

example (a b : ℕ) : a + b = b + a := by ring

example (n : ℕ) (h : n ≠ 0) : n ≥ 1 := by
    cases n with
    | zero => contradiction
    | succ k => exact Nat.succ_pos k

example (n : ℕ) (h : n ≠ 0) : n ≥ 1 := by omega

example (a b : ℕ) : a < b → a ≤ b := by
   intro h
   exact Nat.le_of_lt h

example : ∃ n : ℕ, n + 5 = 8 := by
   use 3

example : ∀ n : ℕ, n + 0 = n := by
   intro n
   exact Nat.add_zero n

example : ∀ n : ℕ, ∃ m : ℕ, m = n + 1 := by
   intro n
   use (n + 1)

example : ∀ a b : ℕ, ∃ c : ℕ, c = a + b := by
   intro a b
   use (a + b)

example (a b c : ℕ) (hab : a ∣ b) (hbc : b ∣ c) : a ∣ c := by
  obtain ⟨k, hk⟩ := hab
  obtain ⟨m, hm⟩ := hbc
  use (k * m)
  calc
     c = b * m := hm
     _ = (a * k) * m := by rw [hk]
     _ = a * (k * m) := by ring

example (a b c : ℕ) (hab : a ∣ b) (hbc : b ∣ c) : a ∣ c :=
   Nat.dvd_trans hab hbc

example (a b d : ℕ) (ha : d ∣ a) (hb : d ∣ b) : d ∣ (a + b) := by
   obtain ⟨k, hk⟩ := ha
   obtain ⟨m, hm⟩ := hb
   use (k + m)
   calc
     a + b = d * k + d * m := by rw [hk, hm]
         _ = d * (k + m)   := by ring

example (a b d : ℕ) (ha : d ∣ a) : d ∣ (a * b) := by
   obtain ⟨k, hk⟩ := ha
   use (k * b)
   calc
      a * b = d * k * b := by rw [hk]
          _ = d * (k * b) := by ring

def ap_term (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d

example (a₁ d : ℕ) : ap_term a₁ d 1 = a₁ := by
   calc
      ap_term a₁ d 1 = a₁ + (1 - 1) * d := by rw [ap_term]
                   _ = a₁ := by ring

lemma my_zero_add (n : ℕ) : 0 + n = n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    calc 0 + Nat.succ n = Nat.succ (0 + n) := rfl
         _ = Nat.succ n := by rw [ih]

lemma my_succ_add (a b : ℕ) : Nat.succ a + b = Nat.succ (a + b) := by
  induction b with
  | zero => rfl
  | succ b ih =>
    calc Nat.succ a + Nat.succ b = Nat.succ (Nat.succ a + b) := rfl
         _ = Nat.succ (Nat.succ (a + b)) := by rw [ih]

-- Now the main theorem
theorem my_add_comm (a b : ℕ) : a + b = b + a := by
  induction b with
  | zero =>
    calc a + 0 = a := Nat.add_zero a
         _ = 0 + a := by rw [my_zero_add]
  | succ b ih =>
    calc a + Nat.succ b = Nat.succ (a + b) := rfl
         _ = Nat.succ (b + a) := by rw [ih]
         _ = Nat.succ b + a := by rw [my_succ_add]

theorem even_square_implies_even (n : ℕ) : Even (n^2) → Even n := by
  intro h_square_even
  by_contra h_not_even

  -- Claim: if n is not even, then n must be odd
  have h_odd : Odd n := by
    -- Every natural number is either even or odd
    cases Nat.even_or_odd n with
    | inl hev =>   -- Case: n is even
      contradiction
    | inr hodd =>  -- Case: n is odd
      exact hodd

  -- Extract witnesses
  obtain ⟨k, hk⟩ := h_odd
  obtain ⟨m, hm⟩ := h_square_even

  -- If n = 2k + 1, then n² is odd
  have h_square_odd : n^2 = 2 * (2*k^2 + 2*k) + 1 := by
    calc n^2 = (2*k + 1)^2 := by rw [hk]
         _ = 4*k^2 + 4*k + 1 := by ring
         _ = 2*(2*k^2 + 2*k) + 1 := by ring

  -- But n² is also even
  rw [hm] at h_square_odd

  -- Contradiction!
  omega

theorem t : Nat.succ (Nat.succ 0) + Nat.succ (Nat.succ 0) = 
            Nat.succ (Nat.succ (Nat.succ (Nat.succ 0))) := by
  rfl



   


