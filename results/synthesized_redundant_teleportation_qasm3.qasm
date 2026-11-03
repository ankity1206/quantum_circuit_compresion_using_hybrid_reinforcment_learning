OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
h q[1];
ry(pi/2) q[0];
cx q[0], q[2];
