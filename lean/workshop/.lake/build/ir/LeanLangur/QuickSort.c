// Lean compiler output
// Module: LeanLangur.QuickSort
// Imports: public import Init public import Mathlib.Data.List.Basic public import Mathlib.Tactic
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
LEAN_EXPORT lean_object* l___private_LeanLangur_QuickSort_0__quickSort_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* l___private_LeanLangur_QuickSort_0__quickSort_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t l_smaller___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_List_filterTR_loop___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_smaller___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* l___private_LeanLangur_QuickSort_0__quickSort_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_smaller___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_larger___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_larger(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_quickSort___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* l___private_LeanLangur_QuickSort_0__quickSort_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_larger___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_quickSort(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_smaller(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t l_larger___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t l_smaller___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_4 = lean_ctor_get(x_1, 4);
lean_inc_ref(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_3, x_2);
x_6 = lean_unbox(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* l_smaller___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_alloc_closure((void*)(l_smaller___redArg___lam__0___boxed), 3, 2);
lean_closure_set(x_4, 0, x_1);
lean_closure_set(x_4, 1, x_2);
x_5 = lean_box(0);
x_6 = l_List_filterTR_loop___redArg(x_4, x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* l_smaller(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = l_smaller___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* l_smaller___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = l_smaller___redArg___lam__0(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT uint8_t l_larger___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_4 = lean_ctor_get(x_1, 6);
lean_inc_ref(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_2, x_3);
x_6 = lean_unbox(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* l_larger___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_alloc_closure((void*)(l_larger___redArg___lam__0___boxed), 3, 2);
lean_closure_set(x_4, 0, x_1);
lean_closure_set(x_4, 1, x_2);
x_5 = lean_box(0);
x_6 = l_List_filterTR_loop___redArg(x_4, x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* l_larger(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = l_larger___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* l_larger___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = l_larger___redArg___lam__0(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* l_quickSort___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_dec_ref(x_1);
return x_2;
}
else
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 1);
lean_inc(x_5);
lean_inc(x_4);
lean_inc_ref(x_1);
x_6 = l_smaller___redArg(x_1, x_4, x_5);
lean_inc_ref(x_1);
x_7 = l_quickSort___redArg(x_1, x_6);
lean_inc(x_4);
lean_inc_ref(x_1);
x_8 = l_larger___redArg(x_1, x_4, x_5);
x_9 = l_quickSort___redArg(x_1, x_8);
lean_ctor_set(x_2, 1, x_9);
x_10 = l_List_appendTR___redArg(x_7, x_2);
return x_10;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_11 = lean_ctor_get(x_2, 0);
x_12 = lean_ctor_get(x_2, 1);
lean_inc(x_12);
lean_inc(x_11);
lean_dec(x_2);
lean_inc(x_12);
lean_inc(x_11);
lean_inc_ref(x_1);
x_13 = l_smaller___redArg(x_1, x_11, x_12);
lean_inc_ref(x_1);
x_14 = l_quickSort___redArg(x_1, x_13);
lean_inc(x_11);
lean_inc_ref(x_1);
x_15 = l_larger___redArg(x_1, x_11, x_12);
x_16 = l_quickSort___redArg(x_1, x_15);
x_17 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_17, 0, x_11);
lean_ctor_set(x_17, 1, x_16);
x_18 = l_List_appendTR___redArg(x_14, x_17);
return x_18;
}
}
}
}
LEAN_EXPORT lean_object* l_quickSort(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = l_quickSort___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* l___private_LeanLangur_QuickSort_0__quickSort_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_dec(x_3);
lean_inc(x_2);
return x_2;
}
else
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 1);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_apply_2(x_3, x_4, x_5);
return x_6;
}
}
}
LEAN_EXPORT lean_object* l___private_LeanLangur_QuickSort_0__quickSort_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = l___private_LeanLangur_QuickSort_0__quickSort_match__1_splitter___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* l___private_LeanLangur_QuickSort_0__quickSort_match__1_splitter___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = l___private_LeanLangur_QuickSort_0__quickSort_match__1_splitter___redArg(x_1, x_2, x_3);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* l___private_LeanLangur_QuickSort_0__quickSort_match__1_splitter___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = l___private_LeanLangur_QuickSort_0__quickSort_match__1_splitter(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_4);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Mathlib_Data_List_Basic(uint8_t builtin);
lean_object* initialize_Mathlib_Tactic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_LeanLangur_QuickSort(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Mathlib_Data_List_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Mathlib_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
