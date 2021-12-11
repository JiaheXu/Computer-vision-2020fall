clc;clear;

%% Question 1
disp("Question 1")
K = [100 0 320;
     0 100 240;
     0 0 1];
R = eye(3);
twp = [0 0 0]';
twpp = [1 0 0]';
P = K*[R' -R'*twp];
Pp = K*[R' -R'*twpp];
disp('P:')
disp(P)
disp("P':")
disp(Pp)
X1 = [0 0 1 1]';
X2 = [1 0 1 1]';
X3 = [1 0 2 1]';
X = [X1 X2 X3];
x = P * X;
xp = Pp * X;
x_inhomo = [];
xp_inhomo = [];
for i=1:3
    x_inhomo(:,end+1) = [x(1,i)/x(3,i) x(2,i)/x(3,i)]';
    xp_inhomo(:,end+1) = [xp(1,i)/xp(3,i) xp(2,i)/xp(3,i)]';
end
disp('x1:')
disp(x_inhomo(:,1))
disp("x1':")
disp(xp_inhomo(:,1))
disp('x2:')
disp(x_inhomo(:,2))
disp("x2':")
disp(xp_inhomo(:,2))
disp('x3:')
disp(x_inhomo(:,3))
disp("x3':")
disp(xp_inhomo(:,3))

disp("xi and xi' with noise:")
N = [-0.1068 0.3714 -1.0891
     1.5326 -0.2256 0.0326];
Np = [1.1006 -1.4916 2.3505;
      1.5442 -0.7423 -0.6156];
x_inhomo = x_inhomo + N;
xp_inhomo = xp_inhomo + Np;
disp('x1:')
disp(x_inhomo(:,1))
disp("x1':")
disp(xp_inhomo(:,1))
disp('x2:')
disp(x_inhomo(:,2))
disp("x2':")
disp(xp_inhomo(:,2))
disp('x3:')
disp(x_inhomo(:,3))
disp("x3':")
disp(xp_inhomo(:,3))

%% Question 2
disp('Question 2')
A1 = [x_inhomo(1,1)*P(3,:)-P(1,:);
      x_inhomo(2,1)*P(3,:)-P(2,:);
      xp_inhomo(1,1)*Pp(3,:)-Pp(1,:);
      xp_inhomo(2,1)*Pp(3,:)-Pp(2,:)];
A2 = [x_inhomo(1,2)*P(3,:)-P(1,:);
      x_inhomo(2,2)*P(3,:)-P(2,:);
      xp_inhomo(1,2)*Pp(3,:)-Pp(1,:);
      xp_inhomo(2,2)*Pp(3,:)-Pp(2,:)];
A3 = [x_inhomo(1,3)*P(3,:)-P(1,:);
      x_inhomo(2,3)*P(3,:)-P(2,:);
      xp_inhomo(1,3)*Pp(3,:)-Pp(1,:);
      xp_inhomo(2,3)*Pp(3,:)-Pp(2,:)];
disp('A1:')
disp(A1)
disp('A2:')
disp(A2)
disp('A3:')
disp(A3)

[U1,S1,V1] = svd(A1);
X1_rec = V1(:,end);
X1_rec_inhomo = [X1_rec(1)/X1_rec(4) X1_rec(2)/X1_rec(4) X1_rec(3)/X1_rec(4)]';
disp('Reconstructed X1:')
disp(X1_rec_inhomo)

[U2,S2,V2] = svd(A2);
X2_rec = V2(:,end);
X2_rec_inhomo = [X2_rec(1)/X2_rec(4) X2_rec(2)/X2_rec(4) X2_rec(3)/X2_rec(4)]';
disp('Reconstructed X2:')
disp(X2_rec_inhomo)

[U3,S3,V3] = svd(A3);
X3_rec = V3(:,end);
X3_rec_inhomo = [X3_rec(1)/X3_rec(4) X3_rec(2)/X3_rec(4) X3_rec(3)/X3_rec(4)]';
disp('Reconstructed X3:')
disp(X3_rec_inhomo)


%% Question 3
disp('Question 3:')
x_new = [50 152 250;
         50 150 248];
A1_3 = [x_new(1,1)*P(3,:)-P(1,:);
      x_new(2,1)*P(3,:)-P(2,:);
      x_new(1,1)*Pp(3,:)-Pp(1,:);
      x_new(2,1)*Pp(3,:)-Pp(2,:)];
A2_3 = [x_new(1,2)*P(3,:)-P(1,:);
      x_new(2,2)*P(3,:)-P(2,:);
      x_new(1,2)*Pp(3,:)-Pp(1,:);
      x_new(2,2)*Pp(3,:)-Pp(2,:)];
A3_3 = [x_new(1,3)*P(3,:)-P(1,:);
      x_new(2,3)*P(3,:)-P(2,:);
      x_new(1,3)*Pp(3,:)-Pp(1,:);
      x_new(2,3)*Pp(3,:)-Pp(2,:)];
% disp('A1 new:')
% disp(A1_3)
% disp('A2 new:')
% disp(A2_3)
% disp('A3 new:')
% disp(A3_3)

[U1_3,S1_3,V1_3] = svd(A1_3);
X1_new = V1_3(:,end);
X1_new = [X1_new(1)/X1_new(4) X1_new(2)/X1_new(4) X1_new(3)/X1_new(4)]';
disp('Infinity point 1:')
disp(X1_new)

[U2_3,S2_3,V2_3] = svd(A2_3);
X2_new = V2_3(:,end);
X2_new = [X2_new(1)/X2_new(4) X2_new(2)/X2_new(4) X2_new(3)/X2_new(4)]';
disp('Infinity point 2:')
disp(X2_new)

[U3_3,S3_3,V3_3] = svd(A3_3);
X3_new = V3_3(:,end);
X3_new = [X3_new(1)/X3_new(4) X3_new(2)/X3_new(4) X3_new(3)/X3_new(4)]';
disp('Infinity point 3:')
disp(X3_new)

X_new = [X1_new' 1; X2_new' 1; X3_new' 1];
[Unew,Snew,Vnew] = svd(X_new);
pi_infinity = Vnew(:,end);
% pi_infinity = null(X_new);
disp('Plane at infinity:')
disp(pi_infinity)
% disp(X_new*pi_infinity)

HA = [eye(3) [0 0 0]';
     pi_infinity'];
disp('HA:')
disp(HA)
XA1 = HA*X1_rec;
XA1_inhomo = [XA1(1)/XA1(4) XA1(2)/XA1(4) XA1(3)/XA1(4)]';
XA2 = HA*X2_rec;
XA2_inhomo = [XA2(1)/XA2(4) XA2(2)/XA2(4) XA2(3)/XA2(4)]';
XA3 = HA*X3_rec;
XA3_inhomo = [XA3(1)/XA3(4) XA3(2)/XA3(4) XA3(3)/XA3(4)]';
PA = P/HA;
PAp = Pp/HA;
disp('XA1:')
disp(XA1_inhomo)
disp('XA2:')
disp(XA2_inhomo)
disp('XA3:')
disp(XA3_inhomo)
disp('PA:')
disp(PA)
disp("PA':")
disp(PAp)



%% Question 4
disp('Question 4:')
w = inv(K*K');
disp('w:')
disp(w)
M = K*R;
AAt = inv(M'*w*M);
A_4 = chol(AAt);
% disp(A_4)
HM = [inv(A_4) [0 0 0]';
      0 0 0 1];
XM1 = HM * XA1;
XM1_inhomo = [XM1(1)/XM1(4) XM1(2)/XM1(4) XM1(3)/XM1(4)]';
XM2 = HM * XA2;
XM2_inhomo = [XM2(1)/XM2(4) XM2(2)/XM2(4) XM2(3)/XM2(4)]';
XM3 = HM * XA3;
XM3_inhomo = [XM3(1)/XM3(4) XM3(2)/XM3(4) XM3(3)/XM3(4)]';
PM = PA/HM;
PMp = PAp/HM;
disp('XM1:')
disp(XM1_inhomo)
disp('XM2:')
disp(XM2_inhomo)
disp('XM3:')
disp(XM3_inhomo)
disp('PM:')
disp(PM)
disp("PM':")
disp(PMp)




