clc;
clear;

%% Question 1
disp('Question 1')
K = [100,0,320;
     0,100,240;
     0,0,1];
Rwc = [-1,0,0;
       0,-1,0;
       0,0,1];
twc = [1,1,-1]';
% disp(-Rwc'*twc)
P = K*[Rwc' -Rwc'*twc];
disp('P:')
disp(P)

X1 = [0,0,5,1]';
X2 = [1,0,7,1]';
X3 = [1,1,8,1]';
X4 = [-6,8,8,1]';
X5 = [2,4,10,1]';
X6 = [-3,8,8,1]';
X = [X1 X2 X3 X4 X5 X6];

x = [];
for i = 1:6
    xi_h = P*X(:,i);
    xi = [xi_h(1)/xi_h(3) xi_h(2)/xi_h(3)]';
    x = [x xi];
end
disp('x:')
disp(x)

%% Question 2
disp('Question 2')
A = [];
for i = 1:6
    Ai = [[0,0,0,0],-X(:,i)',x(2,i)*X(:,i)';
          X(:,i)',[0,0,0,0],-x(1,i)*X(:,i)'];
    A = [A;Ai];
end
disp('A:')
disp(A)
[U, S, V] = svd(A);
p = V(:,end);
disp('p:')
disp(p);

P_DLT_h = [p(1:4)';p(5:8)';p(9:12)'];
P_DLT = P_DLT_h/P_DLT_h(3,4);
disp('P:')
disp(P_DLT)


%% Question 3
disp('Question 3')
B = P_DLT(:,1:3);
M = [0,0,1;0,1,0;1,0,0];
B_f = M*B;
[Q_f, R_f] = qr(B_f');
Q = M*Q_f';
R = M*R_f'*M;
D = eye(3);
for i = 1:3
    D(i,i) = sign(R(i,i));
end

K_3 = R*D;
R_cw_3 = D\Q;
R_wc_3 = R_cw_3';

C = P_DLT(:,4);
t_wc_3 = -R_cw_3*inv(K_3)*C;


disp('Q:')
disp(Q)
disp('R:')
disp(R)
disp('D:')
disp(D)
disp('K:')
disp(K_3)
disp('R_wc:')
disp(R_wc_3)
disp('t_wc:')
disp(t_wc_3)

