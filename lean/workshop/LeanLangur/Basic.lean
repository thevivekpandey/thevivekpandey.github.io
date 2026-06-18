example: And True True :=
   And.intro True.intro True.intro

example : True ∧ True :=
   And.intro True.intro True.intro

#check True


def inc (x : Nat) : Nat := x + 1

def minc : Nat → Nat := fun x => x + 1

def f1 : Nat := 2
def f2 : Nat := 2

#eval (fun  (x : Nat ) => x + 1) 4  

def x :  And True True := 
   And.intro True.intro True.intro

def y: And True False :=
   And.intro True.intro True.intro