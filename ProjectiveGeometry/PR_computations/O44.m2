k = ZZ/101;

-- Our goal is to explicitly compute the projection ramification map
-- for E = O(4)+O(4)

-- The starting point is a map O^4 -> E, which is a 3x2 matrix of
-- quartic forms.  But, modulo the GL(3)xGL(2) action, it can be
-- brought into a normal form (see O33 and O22).

A = k[a_1..a_6,b_1..b_6,c_1..c_6]
S = A[X,Y]
M = matrix ({{X*(a_1*X^3+a_2*X^2*Y+a_3*X*Y^2+Y^3), X*(a_4*X^3+a_5*X^2*Y+a_6*X*Y^2)},
    	     {(X-Y)*(b_1*X^3+b_2*X^2*Y+b_3*X*Y^2), (X-Y)*(b_4*X^3 + b_5*X^2*Y + b_6*X*Y^2 + Y^3)},	
    	     {(X-2*Y)*(X^3+c_1*X^2*Y+c_2*X*Y^2+c_3*Y^3), (X-2*Y)*(X^3+c_4*X^2*Y+c_5*X*Y^2+c_6*Y^3)}})

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

mons = flatten entries gens (ideal(X,Y))^10
dMQmatrix = transpose matrix apply (dMQ, t->flatten entries (coefficients(t, Monomials=>mons))_1)
dMQmatrix = lift(dMQmatrix, A)

-- Finding the degree of the inverse

-- We pick a particular k-valiued point of Gr(2, 11)
-- Given by the kernel of a matrix k^11 -> k^9
P = random(k^9, k^11)

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

-- In Singular
f = "~/Documents/PR/O44.sing"
f << "ring R = 101, " << gensB << ", dp;" << endl << "ideal I = " << "(" << gensI << ");" << endl << "vdim(I)" << endl << close

-- Same computation in MAGMA
f = "~/Documents/PR/O44.magma"
ngensB = toString (length gens A + 1)
f << "k := FiniteField(101);" << endl << "S<" << gensB << "> := PolynomialRing(k, " << ngensB << ");" << endl << " I := ideal< S | " << gensI << " >;" << endl << "VarietySizeOverAlgebraicClosure(I);quit;" << endl << close













