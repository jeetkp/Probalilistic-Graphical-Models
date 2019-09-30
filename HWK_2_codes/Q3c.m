clc;clear all; close all;
% alpha=1.3;
alpha = 0.5;
% beta=1.3;
beta=1.5;
% gamma=1.4;
gamma= 1;
% theta=1.0;
theta= 0.8;
% mu=0.8;
mu= 1.2;

X=[0 0 1; 0 1 1 ; 0 0 1];
Y= [1; 0 ;0 ];
Z=[ 0 1 1 ; 1 1 1 ; 0 0 1];
[m,n]=size(Z)
Zcol=[Z(:,2:n) 2*ones(m,1)];
Zrow=[Z(2:m,:);2*ones(1,n)];
funcAlpha = @(Z) alpha.^(Z==1)
funcBeta = @(Z,Zcol) beta.^(Z==Zcol)
funcGamma = @(Z,Zrow) gamma.^(Z==Zrow)
funcTheta = @(Z,X) theta.^(Z==X)
funcMu = @(X,Y) mu.^(X==1 && Y==1)
% num=1;
% ele=1;
% ele2=1;
% ele3=1;
% for i=1:m
%     for j=1:n
% %         ele=1;
%         ele= ele *funcAlpha(Z(i,j))*funcBeta(Z(i,j),Zcol(i,j))*funcGamma(Z(i,j),Zrow(i,j));
%         for k=1:m
%             for l = 1:n
% %                 ele2=1;
%                 ele2= ele2* funcTheta(Z(i,j),X(k,l));
%                 for ii=1:m
% %                     ele3=1;
%                     ele3= ele3* funcMu(X(k,l),Y(ii));
%                 end
%                 ele2=ele2*ele3;
%                 ele3=1;
%             end
%         end
%         prod=ele*ele2;
%         num=num*prod;
%         ele=1;
%         ele2=1;
% %         ele3=1;
%     end
% end
% num
% clear ele,ele2,ele3;
clear prod;
ele=ones(1,9);
ele2=ones(1,9);
ele1=ones(1,9);
iii=1;
jjj=1;
kkk=1;
for i=1:m
    for j=1:n
        ele(iii)=funcAlpha(Z(i,j))*funcBeta(Z(i,j),Zcol(i,j))*funcGamma(Z(i,j),Zrow(i,j));
        iii=iii+1;
    end
end

for i=1:m
    for j=1:n
%         for k=1:m
%             for l=1:n
                ele1(jjj)=ele1(jjj)*funcTheta(Z(i,j),X(i,j));
%             end
%         end
        jjj=jjj+1;
    end
end

for i=1:m
    for j=1:n
%         for k=1:m
                ele2(kkk)=ele2(kkk)*funcMu(X(i,j),Y(i));
%         end
        kkk=kkk+1;
    end
end
prod1=1;
for i=1:9
    prod1= prod1.*ele(i).*ele1(i)*ele2(i);
end
num = prod(ele)*prod(ele1)*prod(ele2)

den=0;
n=3;
yy=(dec2bin(2^n-1:-1:0)-'0')';
zz=(arrayfun(@(n) logical(bitand(n, [1 2 4; 8 16 32; 64 128 256])) , 0:511, 'UniformOutput', false));

for count1=1:512
    Z=zz{count1};
        Zcol=[Z(:,2:n) 2*ones(m,1)];
        Zrow=[Z(2:m,:);2*ones(1,n)];
        X=[0 0 1; 0 1 1 ; 0 0 1];
    for count2= 1:8        
        Y= yy(:,count2);
ele=ones(1,9);
ele2=ones(1,9);
ele1=ones(1,9);
iii=1;
jjj=1;
kkk=1;
for i=1:m
    for j=1:n
        ele(iii)=funcAlpha(Z(i,j))*funcBeta(Z(i,j),Zcol(i,j))*funcGamma(Z(i,j),Zrow(i,j));
        iii=iii+1;
    end
end

for i=1:m
    for j=1:n
%         for k=1:m
%             for l=1:n
                ele1(jjj)=ele1(jjj)*funcTheta(Z(i,j),X(i,j));
%             end
%         end
        jjj=jjj+1;
    end
end

for i=1:m
    for j=1:n
%         for k=1:m
                ele2(kkk)=ele2(kkk)*funcMu(X(i,j),Y(i));
%         end
        kkk=kkk+1;
    end
end
prod1=1;
for i=1:9
    prod1= prod1.*ele(i).*ele1(i)*ele2(i);
end
 den = den + prod(ele)*prod(ele1)*prod(ele2)
    end
   
end

fprintf(" Alpha = %f \n",alpha);
fprintf(" Beta = %f \n",beta);
fprintf(" Gamma = %f \n",gamma);
fprintf(" Theta = %f \n",theta);
fprintf(" Mu = %f \n",mu);
fprintf(" The output of function is = %f\n", num/den);
