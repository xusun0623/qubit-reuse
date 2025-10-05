OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
cx q[0], q[1];
U(0, 0, pi/3) q[1];
cx q[0], q[1];
