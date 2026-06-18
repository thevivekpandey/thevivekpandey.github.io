import LeanLangur.People
import Mathlib

variable {α β : Type}[LinearOrder α]

def smallestBy (size: β → α)(l: List β)
   (h: l ≠ []) : β := match l with
   | [x] => x
   | x :: y :: xs =>
    let s :=  smallestBy size (y :: xs) (by simp)
    if size x ≤ size s
       then x else s

def youngest (l: List Person)
   (h: l ≠ []) : Person :=
  smallestBy (fun p => p.age) l h

/-!
### Exercises
Prove that `youngest` indeed returns a person from the list who is the youngest.
-/

theorem youngest_in_list (l: List Person) (h: l ≠ []):
   youngest l h ∈ l := 
     match l with
  | [x] => grind
  | x :: y :: xs =>
    have ih := youngest_in_list (y :: xs) (by simp)
    grind

theorem youngest_is_youngest (l: List Person) (h: l ≠ []):
   ∀ x ∈ l ((youngest l h).age ≤ x.age) :=
   sorry

theorem youngest_is_youngest1(l: List Person) (h: l ≠ []) (x : Person):
   x ∈ l → (youngest l h).age ≤ x.age :=
   match l with 
   |[y] => grind
   | y :: z :: xs =>
    have ih :=
      youngest_is_youngest1 (z :: xs) (by simp) x
    grind

