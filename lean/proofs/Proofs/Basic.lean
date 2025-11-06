def hello := "world"

def double (x : Nat): Nat := x * x

def isZero (x : Nat): Bool :=
   if x == 0 then true else false

def sum_up_to (n: Nat) : Nat :=
 match n with
 | Nat.zero => 0
 | Nat.succ n => n + sum_up_to n

def listLength (x : List Nat) : Nat :=
  match x with
  | [] => 0
  | _ :: ys => 1 + listLength ys


def myMap {α β : Type}(f: α → β) (x : List α) : List β :=
   match x with
  | [] => []
  | y :: ys => f y :: myMap f ys

def safeDiv (a: Nat) (b: Nat) : Option Nat :=
   match b with
   | 0 => none
   | _ => some  (a / b)

def replicate {α: Type} (n: Nat) (x: α) : List α :=
   match n with
   | 0 => []
   | Nat.succ k => x :: replicate k x

inductive Color where
  | Red
  | Green
  | Blue

inductive BinaryTree where
  | leaf
  | node (key: Nat) (left: BinaryTree) (right: BinaryTree)

def treeSize (t: BinaryTree): Nat :=
  match t with
  | .leaf => 0
  | .node _ left right => 1 + treeSize left + treeSize right

def treeInsert (t: BinaryTree) (x: Nat): BinaryTree :=
   match t with
  | .leaf => BinaryTree.node x BinaryTree.leaf BinaryTree.leaf
  | .node key left right =>
     if x > key then BinaryTree.node key left (treeInsert right x)
     else BinaryTree.node key (treeInsert left x) right

inductive Vec (α : Type) : Nat → Type where
  | nil : Vec α 0
  | cons : {n: Nat} → α → Vec α n → Vec α (n + 1)

def zipVec {α β : Type} {n: Nat} : Vec α n → Vec β n → Vec (α × β) n
  | Vec.nil, Vec.nil => Vec.nil
  | Vec.cons x xs, Vec.cons y ys => Vec.cons (x, y) (zipVec xs ys)

def zipVec1 {α β : Type} {n : Nat} : Vec α n → Vec β n → Vec (α × β) n :=
  fun v1 v2 =>
    match v1, v2 with
    | Vec.nil, Vec.nil => Vec.nil
    | Vec.cons x xs, Vec.cons y ys => Vec.cons (x, y) (zipVec xs ys)

def head {α : Type} {n : Nat} : Vec α (n + 1) → α :=
   fun v =>
   match v with
   | Vec.cons x _ => x


def head1 {α : Type} {n : Nat} : Vec α (n + 1) → α
   | Vec.cons x _ => x
