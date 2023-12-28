\* Source Pyomo model name=unknown *\

min 
objective:
+ [
+0.2 x(0) ^ 2
+0.04 x(0) * x(1)
+0.08 x(0) * x(2)
+0.16 x(1) ^ 2
+0.04 x(1) * x(2)
+0.14 x(2) ^ 2
+0.04 x(3) ^ 2
] / 2

s.t.

c_l_return_constraint_:
+0.12 x(0)
+0.1 x(1)
+0.07 x(2)
+0.03 x(3)
>= 0.08

c_e_sum_constraint_:
+1 x(0)
+1 x(1)
+1 x(2)
+1 x(3)
= 1

bounds
   0 <= x(0) <= +inf
   0 <= x(1) <= +inf
   0 <= x(2) <= +inf
   0 <= x(3) <= +inf
end
