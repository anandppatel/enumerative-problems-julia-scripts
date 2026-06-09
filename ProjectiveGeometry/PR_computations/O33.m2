k = ZZ/101;

-- Our goal is to explicitly compute the projection ramification map
-- for E = O(3)+O(3)

-- The starting point is a map O^4 -> E, which is a 3x2 matrix of
-- cubic forms.  But, modulo the GL(3)xGL(2) action, it can be
-- brought into the following normal form.

--   X 	   [a1*X^2 + a2*XY + Y^2, a3*X^2+a4*X*Y]
--   (X-Y) [b1*X^2+b2*X*Y, b3*X^2 + b4*X*Y + Y^2]
--   (X-2Y)[X^2+c1*X*Y+c2*Y^2, X^2+c3*X*Y+c4*Y^2]

A = k[a_1..a_4,b_1..b_4,c_1..c_4]
S = A[X,Y]
M = matrix ({{X*(a_1*X^2+a_2*X*Y+Y^2), X*(a_3*X^2+a_4*X*Y)},
    	     {(X-Y)*(b_1*X^2+b_2*X*Y), (X-Y)*(b_3*X^2 + b_4*X*Y + Y^2)},	
    	     {(X-2*Y)*(X^2+c_1*X*Y+c_2*Y^2), (X-2*Y)*(X^2+c_3*X*Y+c_4*Y^2)}})

-- The next task is to compute the 2x2 minors of this matrix.
Q = gens minors(2, M)
--- The list of minors is the one given by subsets(3,2)
-- {{0, 1}, {0, 2}, {1, 2}}
-- We want the exact opposite order.
-- Also, the minors do not have the right signs.
Q = matrix {apply (3, (i -> (reverse flatten entries Q)_i * (-1)^i))}

-- We check that Q * M == 0
assert (Q * M == 0)

-- To compute the differential map, we take X-derivatives of each entry of Q
dQ = matrix {apply (flatten entries Q, t -> diff(X, t))}

-- Now we take dQ * M. Its entries must be divisible by Y
predMQ = dQ * M
scan (flatten entries predMQ, t -> assert (t % Y == 0))

dMQ = apply (flatten entries predMQ, t -> t // Y)

-- We now extract the coefficients, and interpret the result as
-- vectors in a 8 dim vector space.

mons = flatten entries gens (ideal(X,Y))^7
dMQmatrix = transpose matrix apply (dMQ, t->flatten entries (coefficients(t, Monomials=>mons))_1)
dMQmatrix = lift(dMQmatrix, A)

-- Finding the degree of the inverse

-- We pick a particular k-valiued point of Gr(2, 8)
-- Given by the kernel of a matrix k^8 -> k^6
P = random(k^6, k^8)

-- We promote it to a map A^8 -> A^6
PA = promote(P, A)

-- We now insist that the point given by dMQmatrix be the same as the
-- one given by PA. To do this, it suffices to impose the conditions that
-- (1) dMQmatrix has full rank and 
-- (2) the composite PA * dMQmatrix is zero.

-- To impose the first condition, we invert the first 2x2 minor of dMQmatrix.

firstMinor = det(submatrix(transpose dMQmatrix, {0,1}), Strategy => Cofactor);

B = A[t]
I = promote(ideal (PA * dMQmatrix), B)  + ideal(firstMinor*t - 1)

gensI = toString flatten entries gens I
gensI = substring(1,#gensI-2,gensI)
gensB = toString gens A 
gensB = concatenate(substring(1, #gensB-2, gensB), ", t")

f = "~/Documents/PR/O33.sing"
f << "ring R = 101, " << gensB << ", dp;" << endl << "ideal I = " << "(" << gensI << ");" << endl << "vdim(I)" << endl << close

-- colength of I turns out to be 22 most of the time.

-- Same computation in MAGMA
f = "~/Documents/PR/O33.magma"
ngensB = toString (length gens A + 1)
f << "k := FiniteField(101);" << endl << "S<" << gensB << "> := PolynomialRing(k, " << ngensB << ");" << endl << " I := ideal< S | " << gensI << " >;" << endl << "VarietySizeOverAlgebraicClosure(I);quit;" << endl << close

--22







