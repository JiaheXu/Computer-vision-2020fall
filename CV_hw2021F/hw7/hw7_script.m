clc;
clear;

%% Question 1
disp('Question 1')
K = [100, 0, 320;
     0, 100, 240;
     0, 0, 1];
Rwc1 = eye(3);
twc1 = [1, 0, 0]';
Rwc2 = ROTY(pi/2);
twc2 = [0, 0, 1]';
P1 = K*[Rwc1' -Rwc1'*twc1];
P2 = K*[Rwc2' -Rwc2'*twc2];
disp('P:')
disp(P1)
disp("P':")
disp(P2)
X = [1, 0, 1, 1]';
x1 = P1*X;
x1 = [x1(1)/x1(3) x1(2)/x1(3) 1]';
x2 = P2*X;
x2 = [x2(1)/x2(3) x2(2)/x2(3) 1]';
disp('x:')
disp(x1)
disp("x':")
disp(x2)
C1 = null(P1);
e2 = P2*C1;
e2 = [e2(1)/e2(3) e2(2)/e2(3) 1]';
disp("e':")
disp(e2)

%% Question 2
disp('Question 2')
P_plus = pinv(P1);
F = SKEW3(e2)*P2*P_plus;
F = F/F(3,3);
disp('F:')
disp(F)
a = x2'*F*x1;
disp("Validate F with x'T*F*x=0")
disp("x'T*F*x:")
disp(a)
b = F'*e2;
disp("Validate F with FT*e'=0")
disp("FT*e':")
disp(b)

%% Question 3
disp('Question 3')
Pc1 = [eye(3) [0,0,0]'];
P1_star = [P1;[0 0 0 1]];
H = inv(P1_star);
disp('H:')
disp(H)
% disp('P1*H:')
% disp(P1*H)
Pc2 = P2*H;
disp("Pc':")
disp(Pc2)
X_new = H\X;
x3 = Pc1*X_new;
x3 = [x3(1)/x3(3) x3(2)/x3(3) 1]';
disp('x:')
disp(x3)
x4 = Pc2*X_new;
x4 = [x4(1)/x4(3) x4(2)/x4(3) 1]';
disp("x':")
disp(x4)
e2_new = Pc2*(H\C1);
e2_new = [e2_new(1)/e2_new(3) e2_new(2)/e2_new(3) 1]';
disp("e':")
disp(e2_new)

%% Question 4
disp('Question 4')
PF1 = [eye(3) [0 0 0]'];
% v = [1 2 3]';
% lamda = 5;
e2_star = null(F');
e2_star = [e2_star(1)/e2_star(3) e2_star(2)/e2_star(3) 1]';
disp("e' by null(F):")
disp(e2_star)
PF2 = [SKEW3(e2_star)*F e2_star];
disp('PF:')
disp(PF1)
disp("PF':")
disp(PF2)
Pc1_plus = pinv(Pc1);
Fc = SKEW3(e2)*Pc2*Pc1_plus;
Fc = Fc/Fc(3,3);
PF1_plus = pinv(PF1);
Ff = SKEW3(e2_star)*PF2*PF1_plus;
Ff = Ff/Ff(3,3);
disp("F calculated by (Pc,Pc'):")
disp(Fc)
disp("F calculated by (PF,PF'):")
disp(Ff)
disp("F of Question 1:")
disp(F)




