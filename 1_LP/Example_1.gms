*Variable declaration
Variable
     x1   Number of comedy ads purchased
     x2   Number of football ads purchased
      
     z    Objective function value
     
;

*Objective function defitition
Equation     obj    Obejective function;
             obj..  z =e= 50*x1 + 100*x2;
             
*Contraint definition
Equation     eq1    Constraint 1;
             eq1..  7*x1 + 2*x2 =g= 28;
             
Equation     eq2    Constaint 2;
             eq2..  2*x1 + 12*x2 =g= 24;
             
*Model assemly
model example_1 /all/;

*Solver option
option lp = cplex;

*Solve statement
solve example_1 using lp minimization z;

