
import Mathlib.tactic
set_option linter.style.commandStart false

inductive myVector (α : Type) : Nat → Type where
  | nil  : myVector α 0
  | cons : {n: Nat} → α → myVector α n → myVector α (n + 1)

#check myVector.nil (α := Nat)

#check myVector.nil

def v : myVector Nat 3 :=
  myVector.cons 1 (myVector.cons 2 (myVector.cons 3 myVector.nil))

def u : myVector Nat 0 := myVector.nil

def n0 : myVector Nat 0 := @myVector.nil Nat
#check n0

def n1 : myVector Nat 1 := myVector.cons 42 n0
def n1a : myVector Nat 1 := myVector.cons 42 (myVector.nil)

def n2 : myVector Nat 2 := myVector.cons 43 n1
def n2a : myVector Nat 2 := myVector.cons 43 (myVector.cons 42 n0)
def n2b : myVector Nat 2 := myVector.cons 43 (myVector.cons 42 (myVector.nil))

#check Prod

universe u v w
structure myPair (α : Type u) (β : Type v) where
  mk ::
  fst : α
  snd : β


structure myTriple (α : Type u) (β : Type v) (γ : Type w) : Type max u (max v w) where
   mk ::
   fst: α
   snd: β
   thrd: γ

#check myTriple
def x1 : myTriple Nat Nat Nat := myTriple.mk 4 5 6
def x3 : myTriple Nat Nat Nat := ⟨4, 5, 6⟩
def x4 : myTriple Nat Nat Nat := {fst := 4, snd := 5, thrd:= 6}

structure myAnd (x : Prop) (y : Prop) where
   mk ::
   fst : x
   snd : y

#check True

def y1 : And True True := And.intro True.intro True.intro

def y2 : And True (And True True) := And.intro True.intro y1

#check Iff

def z1 : True ↔ True := ⟨fun h => h, fun h => h⟩
def z2 : True ↔ True := ⟨fun _ => True.intro, fun _ => True.intro⟩

def addCom (a b : Prop) (_ : a) (_ : b) : And a b ↔ And b a :=
   ⟨fun h => And.intro h.right h.left,
    fun h => And.intro h.right h.left⟩

#check Sigma

def w1 : Sigma (fun n => myVector Nat n) :=
   ⟨3, myVector.cons 1 (myVector.cons 2 (myVector.cons 3 myVector.nil))⟩

def s4 : Sigma (fun α : Type => α) := ⟨Nat, 42⟩

def s5 : Sigma (fun α : Type => α) := ⟨Bool, true⟩

def x : Type := Fin 5

def y : Fin 5 := 6

#eval (9: Fin 6).val

def y3 : Fin 6 := ⟨4, Nat.le.step Nat.le.refl⟩
def y4 : Fin 6 := ⟨4, by omega⟩

/-structure Printable (α : Type) where
   print : α → String

def printNat : Printable Nat := ⟨fun n => toString n⟩
def display (p : Printable Nat) (n: Nat) : String := p.print n

#eval display printNat 42
-/
class Printable (α : Type) where
   print : α → String

instance : Printable Nat where
   print := fun n => toString n

def display {α : Type} [Printable α] (n : α) : String := Printable.print n
#eval display 42

#check Nat.add
#check Float.add
#check Int.add


def allFortyTwo : (n: Nat) → myVector Nat n
   | 0 => myVector.nil
   | n + 1 => myVector.cons 42 (allFortyTwo n)

def allFortyTwo1 (n : Nat) : myVector Nat n :=
  match n with
  | 0     => myVector.nil
  | n + 1 => myVector.cons 42 (allFortyTwo1 n)

theorem ex1 : True := True.intro

theorem ex2 (p : Prop) (h : p) : p := h

theorem ex3 (p q : Prop) (hp : p) (hq : q) : And p q :=
   And.intro hp hq

theorem ex4 (p q : Prop) (h : And p q) : p :=
   h.left

theorem ex5 (p q : Prop) (h : And p q) : And q p :=
   And.intro h.right h.left

theorem ex6 (p q : Prop) (_ : p) (_ : q) : Iff p p:=
   Iff.intro
   (fun (x: p) => (x: p))
   (fun (x: p) => (x: p))

theorem ex7 (p q : Prop) (h : p → q) (hp : p) :q  :=
   h hp

theorem ex8 (p q r : Prop) (hpq : p → q) (hqr : q → r) (hp : p) : r :=
  hqr (hpq hp)

/--theorem ex9 (p q : Prop) (h: And p q) : And q p :=
   And.intro
   h.right
   h.left--/

theorem ex9 (p q : Prop) (h : And p q) : And q p := by exact And.intro h.right h.left

/--theorem ex10 (p q: Prop) (h: And p q) : And q p := by
  constructor
  . exact h.right
  . exact h.left

theorem ex11 (p q: Prop) (h: And p q): And q p := by
  cases h with
  | intro left right =>
     constructor
     .exact right
     .exact left--/

theorem ex10 (p q r : Prop) (hpq : p → q) (hqr : q → r) : p → r := by
   exact (fun (x: p) =>  hqr (hpq x) )

theorem ex11 (p q : Prop) (hpq : p → q) (hqp : q → p) : Iff p q := by
   exact Iff.intro hpq hqp

theorem ex12 (p : Prop) (h : p) : ¬ ¬ p := by
   exact fun hnp => hnp h

theorem ex13 (p : Prop) (h : p) : ¬ ¬ ¬ ¬ p :=
   fun h3 => h3 (fun hnp => hnp h)

theorem ex14 (p : Prop) (h : p) : ¬ ¬ ¬ ¬ ¬ ¬ p :=
   fun h5 => h5 (fun h3 => h3 (fun hnp => hnp h))

theorem ex15 : False → False :=
  fun h => h

theorem ex16 (p : Prop) : False → p :=
   fun h => False.elim h

#check Bool.rec

def myBoolRec {motive : Bool → Sort u}
              (mFalse : motive false)
              (mTrue : motive true)
              (b : Bool) : motive b :=
   match b with
  | false => mFalse
  | true => mTrue

def test (b : Bool) : Nat := myBoolRec (motive:= fun _ => Nat) 4 5 b
#eval test false

def myNatRec {motive : Nat → Sort u}
             (mZero : motive Nat.zero)
             (mSucc : (k : Nat) → motive k → motive (Nat.succ k))
             (n : Nat) : motive n :=
  match n with
  | Nat.zero => mZero
  | Nat.succ k => mSucc k (myNatRec mZero mSucc k)

def add (m n : Nat) : Nat :=
   match m with
  | Nat.zero => n
  | Nat.succ k => Nat.succ (add k n)

def add1 (m n : Nat) : Nat :=
   Nat.rec
   n
   (fun _ s => Nat.succ s)
   m

#eval add1 3 4


theorem zero_add1 (n : Nat) : 0 + n = n :=
   match n with
   | Nat.zero => rfl
   | Nat.succ k => by rw [Nat.add_succ, zero_add1 k]

theorem zero_add2 (n : Nat) : 0 + n = n :=
   match n with
   | Nat.zero => rfl
   | Nat.succ k => congrArg Nat.succ (zero_add2 k)

theorem zero_add3 (n : Nat) : 0 + n = n :=
   match n with
   | Nat.zero => rfl
   | Nat.succ k => by rw [Nat.add_succ, zero_add3 k]

#check Nat.add_succ

theorem add_comm1 (m n : Nat) : m + n = n + m :=
   match n with
   | Nat.zero => by rw [Nat.add_zero, zero_add1]
   | Nat.succ k => by rw [Nat.add_succ, add_comm m k, Nat.succ_add]

theorem add_assoc1 (m n k : Nat) : m + (n + k) = (m + n) + k :=
   match k with
   | Nat.zero => by simp
   | Nat.succ k => by rw [Nat.add_succ, Nat.add_succ, add_assoc1 m n k, Nat.add_succ]

theorem succ_eq_add_one (n : Nat) : Nat.succ n = n + 1 := rfl

theorem mul_zero (n : Nat) : n * 0 = 0 := rfl

theorem mul_succ (m n : Nat) : m * Nat.succ n = m * n + m := rfl

theorem zero_mul (n : Nat) : 0 * n = 0 := by simp

theorem mul_comm1 (m n : Nat) : m * n = n * m :=
  match m with
  | 0 => by simp
  | Nat.succ k => by
    rw [Nat.succ_eq_add_one, Nat.left_distrib, Nat.right_distrib, mul_comm k n]
    have h1: 1 * n = n := by rw [Nat.one_mul]
    have h2: n * 1 = n := by rw [Nat.mul_one]
    rw [h1, h2]

theorem mul_com1 (m n : Nat) : m * n = n * m :=
   match m with
   | 0 => by simp
   | Nat.succ k => by
      rw [Nat.succ_eq_add_one, Nat.left_distrib, Nat.right_distrib]
      rw [mul_comm k n]
      simp

theorem mul_add1 (m n k : Nat) : m * (n + k) = m * n + m * k :=
   match k with
   | 0 => by simp
   | Nat.succ k => calc
      m * (n + Nat.succ k)
        = m * Nat.succ (n + k) := by rw [Nat.add_succ]
      _ = m * (n + k) + m      := by rw [mul_succ]
      _ = (m * n + m * k) + m  := by rw [mul_add m n k]
      _ = m * n + (m * k + m)   := by rw [Nat.add_assoc]
      _ = m * n + m * Nat.succ k := by rw [← Nat.mul_succ]

theorem add_mul1 (m n k : Nat) : (m + n) * k = m * k + n * k :=
   match m with
  | 0 => by simp
  | Nat.succ m => calc
     (Nat.succ m + n) * k
      = (n + Nat.succ m) * k := by rw[Nat.add_comm]
    _ = Nat.succ (n + m) * k := by rw[Nat.add_succ]
    _ = Nat.succ (m + n) * k := by rw[Nat.add_comm]
    _ = (m + n) * k + k      := by rw[Nat.succ_mul]
    _ = (m * k + n * k) + k  := by rw[add_mul]
    _ = m * k + (n * k + k) := by rw[Nat.add_assoc]
    _ = m * k + (k + n * k) := by rw[Nat.add_comm k (n * k)]
    _ = (m * k + k) + n * k := by rw[Nat.add_assoc]
    _ = (Nat.succ m) * k + n * k := by rw[Nat.succ_mul]

theorem mul_assoc1 (m n k : Nat) : (m * n) * k = m * (n * k) :=
  match k with
  | 0 => by simp
  | Nat.succ k => calc
     (m * n) * Nat.succ k
     = (m * n) * k + (m * n) := by rw[Nat.mul_succ]
   _ = m * (n * k) + (m * n) := by rw[mul_assoc]
   _ = m * ((n * k) + n) := by rw[Nat.left_distrib]
   _ = m * (n * Nat.succ k) := by rw[Nat.mul_succ]

def divides (m n : Nat) : Prop := ∃ k, n = m * k

theorem divides_refl (n : Nat) : divides n n :=
   ⟨1, by ring⟩

theorem divides_refl1 (n : Nat) : divides n n := by
   exact ⟨1, by simp⟩

theorem divides_refl2 (n : Nat) : divides n n := by
   use 1
   simp

theorem divides_refl3 (n : Nat) : divides n n :=
   ⟨1, (Nat.mul_one n).symm⟩

theorem divides_refl4 (n : Nat) : divides n n :=
   Exists.intro 1 (Nat.mul_one n).symm

theorem divides_zero1 (n : Nat) : divides n 0 :=
   ⟨0, by simp⟩

theorem divides_zero2 (n : Nat) : divides n 0 := by
   exact ⟨0, by simp⟩

theorem divides_zero3 (n : Nat) : divides n 0 := by
   use 0
   simp

theorem divides_trans (m n p : Nat) (h1 : divides m n) (h2 : divides n p) : divides m p := by
  obtain ⟨k1, hk1⟩ := h1
  obtain ⟨k2, hk2⟩ := h2
  use k1 * k2
  rw [hk2, hk1, mul_assoc]

def isPrime (p : Nat) : Prop :=
  p ≥ 2 ∧ ∀ m, divides m p → m = 1 ∨ m = p

theorem two_is_prime : isPrime 2 := by
  unfold isPrime divides
  constructor
  · omega
  · intro m hm
    obtain ⟨k, hk⟩ := hm
    have hk_pos : k ≥ 1 := by
      cases k with
      | zero => simp at hk
      | succ k => omega
    have hm_le : m ≤ 2 := by nlinarith
    interval_cases m <;> omega

theorem prime_greater_than_one (p : Nat) (hp : isPrime p) : p ≥ 2 := by
   obtain ⟨l, r⟩ := hp
   exact l

theorem prime_greater_than_one1 (p : Nat) (hp : isPrime p) : p ≥ 2 :=
  hp.1

theorem divides_add (m n k : Nat) (h1 : divides m n) (h2 : divides m k) : divides m (n + k) := by
  obtain ⟨k1, hk1⟩ := h1
  obtain ⟨k2, hk2⟩ := h2
  use k1 + k2
  rw [hk1, hk2, Nat.left_distrib]

def factorial : Nat → Nat
  | 0 => 1
  | Nat.succ n => Nat.succ n * factorial n

def smallestDivisorHelper (n k fuel : Nat) : Nat :=
  match fuel with
  | 0 => n
  | Nat.succ f =>
    if k * k > n then n
    else if k ∣ n then k
    else smallestDivisorHelper n (k + 1) f

def smallestDivisor (n : Nat) : Nat :=
  smallestDivisorHelper n 2 n

#check Nat.add_assoc
#print axioms Nat.add_assoc


