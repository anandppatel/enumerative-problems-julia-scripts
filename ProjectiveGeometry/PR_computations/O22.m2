k = ZZ/101;

-- Our goal is to explicitly compute the projection ramification map
-- for E = O(2)+O(2)

-- The starting point is a map O^3 -> E, which is a 3x2 matrix of
-- quadratic forms.  But, modulo the GL(3)xGL(2) action, it can be
-- brought into the following normal form.

--   X 	   [a1*X + Y, a2*X]
--   (X-Y) [b1*X, b2*X + Y]
--   (X-2Y)[X+c1*X, X+c2*X]

A = k[a_1..a_2,b_1..b_2,c_1..c_2]
S = A[X,Y]
M = matrix ({{X*(a_1*X+Y), X*(a_2*X)},
    	     {(X-Y)*b_1*X, (X-Y)*(b_2*X+Y)},	
    	     {(X-2*Y)*(X+c_1*Y), (X-2*Y)*(X+c_2*Y)}})

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
-- vectors in a 7 dim vector space.

mons = flatten entries gens (ideal(X,Y))^4
dMQmatrix = transpose matrix apply (dMQ, t->flatten entries (coefficients(t, Monomials=>mons))_1)
dMQmatrix = lift(dMQmatrix, A)

-- Finding the degree of the inverse

-- We pick a particular k-valiued point of Gr(2, 5)
-- Given by the kernel of a matrix k^5 -> k^3
P = random(k^3, k^5)

-- We promote it to a map A^5 -> A^3
PA = promote(P, A)

-- We now insist that the point given by dMQmatrix be the same as the
-- one given by PA. To do this, it suffices to impose the conditions that
-- (1) dMQmatrix has full rank and 
-- (2) the composite PA * dMQmatrix is zero.

-- To impose the first condition, we invert the first 2x2 minor of dMQmatrix.

firstMinor = det(submatrix(transpose dMQmatrix, {0,1}), Strategy => Cofactor);

B = A[t]
I = promote(ideal (PA * dMQmatrix), B)  + ideal(firstMinor*t - 1)

-- dim I = 0
-- degree I = 2

-- As found before!

gensI = toString flatten entries gens I
gensI = substring(1,#gensI-2,gensI)
gensB = toString gens A 
gensB = concatenate(substring(1, #gensB-2, gensB), ", t")

-- In Singular
f = "~/Documents/PR/O22.sing"
f << "ring R = 101, " << gensB << ", dp;" << endl << "ideal I = " << "(" << gensI << ");" << endl << "vdim(I)" << endl << close

-- Same computation in MAGMA
f = "~/Documents/PR/O22.magma"
ngensB = toString (length gens A + 1)
f << "k := FiniteField(101);" << endl << "S<" << gensB << "> := PolynomialRing(k, " << ngensB << ");" << endl << " I := ideal< S | " << gensI << " >;" << endl << "VarietySizeOverAlgebraicClosure(I);quit;" << endl << close











