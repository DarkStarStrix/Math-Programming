\* Source Pyomo model name=unknown *\

min 
x1:
+3 x2
+3.5 x3
+5 x4
+5.5 x5
+4 x6
+4.5 x7
+3 x8
+3.5 x9

s.t.

c_u_x10_:
+1 x2
+1 x4
<= 100

c_u_x11_:
+1 x3
+1 x5
<= 150

c_u_x12_:
+1 x6
+1 x8
<= 200

c_u_x13_:
+1 x7
+1 x9
<= 100

c_e_x14_:
+1 x2
+1 x6
= 80

c_e_x15_:
+1 x3
+1 x7
= 100

c_e_x16_:
+1 x4
+1 x8
= 120

c_e_x17_:
+1 x5
+1 x9
= 150

bounds
   0 <= x2 <= +inf
   0 <= x3 <= +inf
   0 <= x4 <= +inf
   0 <= x5 <= +inf
   0 <= x6 <= +inf
   0 <= x7 <= +inf
   0 <= x8 <= +inf
   0 <= x9 <= +inf
end
