*Variable definition
Variable
      x1 Number of worker begninning work on Monday
      x2 Number of worker begninning work on Tuesday
      x3 Number of worker begninning work on Wednesday
      x4 Number of worker begninning work on Thursday
      x5 Number of worker begninning work on Friday
      x6 Number of worker begninning work on Saturday
      x7 Number of worker begninning work on Sunday
      
      z Objective fnction value
;

*Objective function definition
Equation    obj     Objective function;
            obj..   z =e= x1 + x2 + x3 + x4 + x5 + x6 + x7;
            
*Constraint definition
Equation    eq1     Monday requiremnet;
            eq1..   x1 + x4 + x5 + x6 + x7 =g= 17;
            
Equation    eq2     Tuesday requiremnet;
            eq2..   x1 + x2 + x5 + x6 + x7 =g= 13;
            
Equation    eq3     Wednesday requiremnet;
            eq3..   x1 + x2 + x3 + x6 + x7 =g= 15;
            
Equation    eq4     Thursday requiremnet;
            eq4..   x1 + x2 + x3 + x4 + x7 =g= 19;
            
Equation    eq5     Friday requiremnet;
            eq5..   x1 + x2 + x3 + x4 + x5 =g= 14;
            
Equation    eq6     Saturday requiremnet;
            eq6..   x2 + x3 + x4 + x5 + x6 =g= 16;
            
Equation    eq7     Sunday requiremnet;
            eq7..   x3 + x4 + x5 + x6 + x7 =g= 11;
            
*Adding non-negativity constraints
Equation    eq8        Non Monday;
            eq8..      x1 =g= 0;
            
Equation    eq9        Non Tuesday;
            eq9..      x2 =g= 0;
            
Equation    eq10       Non Wednesday;
            eq10..     x3 =g= 0;
            
Equation    eq11       Non Thursday;
            eq11..     x4 =g= 0;
            
Equation    eq12       Non Friday;
            eq12..     x5 =g= 0;
            
Equation    eq13       Non Saturday;
            eq13..      x6 =g= 0;
            
Equation    eq14       Non Sunday;
            eq14..     x7 =g= 0;
            
*Model assembly
model example_2 /all/;

option lp =cplex;

solve example_2 using lp minimization z;
            
