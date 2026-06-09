k = ZZ/101;

-- Our goal is to explicitly compute the projection ramification map
-- for E = O(3)+O(5)

-- The starting point is a map O^4 -> E, which is a 3x2 matrix of
-- cubic/quintic forms.  But, modulo the GL(3)x Aut(E) action, it can be
-- brought into a normal form.

A = k[a_1..a_4,b_1..b_7,c_1..c_6]
S = A[X,Y]
M = matrix ({{X*(a_1*X^2+a_2*X*Y+Y^2), X*(a_3*X^4+a_4*X^3*Y)},
    	     {(X-Y)*(b_1*X^2+b_2*X*Y+b_3*Y^2), (X-Y)*(b_4*X^4+b_5*X^3*Y+b_6*X^2*Y^2+b_7*X*Y^3+Y^4)},	
    	     {(X-2*Y)*(c_1*X^2+c_2*X*Y+Y^2), (X-2*Y)*(c_3*X^4+c_4*X^3*Y+c_5*X^2*Y^2+c_6*X*Y^3+Y^4)}})

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

-- The result is a global section of O(9)+O(11), which we have to take modulo Aut.
monsSmaller = flatten entries gens (ideal(X,Y))^9
monsBigger = flatten entries gens (ideal(X,Y))^11
monsDiff = flatten entries gens (ideal(X,Y))^2

-- The smaller is unique (up to scaling), so we store it.
smaller = transpose flatten (coefficients(dMQ_0, Monomials=>monsSmaller))_1
smaller = lift(smaller, A)
-- The bigger is only unique modulo the smaller.
dMQbigger = append(apply(monsDiff, (m -> promote(m,S) * dMQ_0)), dMQ_1)
bigger = transpose matrix apply (dMQbigger, t -> flatten entries (coefficients(t, Monomials => monsBigger))_1)
bigger = lift(bigger, A)

-- Finding the degree of the inverse

-- We pick a particular k-valiued point of P9 + P11 modulo P9
kXY = k[X,Y]
imSmaller = random(9,kXY)
imBigger = random(11, kXY)
monsSmaller = flatten entries gens (ideal(X,Y))^9
monsBigger = flatten entries gens (ideal(X,Y))^11
monsDiff = flatten entries gens (ideal(X,Y))^2
imS = transpose gens kernel transpose lift((coefficients(imSmaller, Monomials=>monsSmaller))_1, k)
imB = transpose gens kernel lift (matrix (append (apply(monsDiff, (m-> flatten entries (coefficients(imSmaller * m, Monomials=>monsBigger))_1)), 
	    flatten entries (coefficients(imBigger, Monomials=>monsBigger))_1)), k)

assert (rank imS == numrows imS)
assert (rank imB == numrows imB)

-- We promote both to A
imSA = promote(imS, A)
imBA = promote(imB, A)

-- We now insist that the point given by dMQmatrix be the same as the
-- one given by PA. imageSmallerA and imageBiggerA
firstMinorSmaller = smaller_(0,0);
lastMinorBigger = det(submatrix(transpose bigger, (numrows bigger) - (numcols bigger) .. (numrows bigger - 1)), Strategy=>Cofactor);

B = A[t_1,t_2]
I = promote(ideal (imBA * bigger), B)  + promote(ideal (imSA * smaller), B) + ideal(firstMinorSmaller*t_1 - 1) + ideal(lastMinorBigger*t_2 - 1);

gensI = toString flatten entries gens I;
gensI = substring(1,#gensI-2,gensI);
gensB = toString gens A ;
gensB = concatenate(substring(1, #gensB-2, gensB), ", t_1, t_2");


f = "~/Documents/PR/O35.sing"
f << "ring R = 101, (" << gensB << "), dp;" << endl << "ideal I = " << "(" << gensI << ");" << endl << "vdim(I)" << endl << close

-- Same computation in MAGMA
f = "~/Documents/PR/O35.magma"
ngensB = toString (length gens A + 2)
f << "k := FiniteField(101);" << endl << "S<" << gensB << "> := PolynomialRing(k, " << ngensB << ");" << endl << " I := ideal< S | " << gensI << " >;" << endl << "VarietySizeOverAlgebraicClosure(I);quit;" << endl << close
--380










