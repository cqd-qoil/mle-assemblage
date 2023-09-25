function [MTestResult, Assem] = problem_5_faster(bobData, Assem)
yalmip('clear');
%Here we add a patch that fixes a bug when calculating the multinomial test value.
%We first try fixing all the scalars while optimizing the matrices. Using
%brute force to search for the change in gamma_x and epsx
% You need to run problem_2_faster to obtain the variable Assem. Then run
% this.
kMax=1;%meaning that Charlie is deleted. If it is 2, Charlie is recovered.

%Install MOSEK: https://docs.mosek.com/latest/install/installation.html
%I use 4.2.3 Windows, MSI installer.
%Then I go to https://www.mosek.com/products/academic-licenses/ to require
%a Personal Academic License.
%Now use https://docs.mosek.com/latest/install/installation.html 4.3 Setting up the License
%restart MATLAB and run mosekdiag. If unsuccessful, maybe need to add MOSEK
%installed file to MATLAB system path.
%Now we can run a test example in MOSEK in
%https://docs.mosek.com/latest/toolbox/design.html#hello-world-in-mosek for
%a test.

%Then Use https://yalmip.github.io/tutorial/installation/   to install and test
%Yalmip.

%Important variables: gamma,epsx,Assem,Tfinal. You can find their meaning
%in each of their first-appearance place.


ops = sdpsettings('solver', 'mosek', 'verbose', 0);%this tells yalmip to use mosek

%--------you need to set the following values manually-     -----%
UseWhich=zeros(1,size(1,2));
UseWhich(1)=1;

LastTwoSum=1;%1 means the last two columns of the data are the total measurement shot numbers for each POVM group; 0 means instead the last two columns are the null results

%--------------------------------%

data=zeros(9,6,size(1,2),2);%the last(fourth) index means AB or AC. So it is with NEG, f, p, N, M, etc.
%9:Alice 3 povms * Bob 3 povms; 6:outcomes, hh hv vh vv nullH nullV;
%8:different alpha_n; 2: Bob/Charlie

data(:,:,1,1)= bobData;


if LastTwoSum~=0
    for n=1:size(data,3)
        for k=1:2
            data(:,5:6,n,k)=data(:,5:6,n,k)-data(:,1:2,n,k)-data(:,3:4,n,k);
        end
    end
end

%-----------the above are loading data and initial commands---------%


I=eye(2);
sigma=zeros(2,2,3);%single-qubit Pauli matrices
sigma(:,:,1)=[1,0;0,-1];
sigma(:,:,2)=[0,1;1,0];
sigma(:,:,3)=[0,-1i;1i,0];
sigma(:,:,4)=eye(2);

NAlpha=size(data,3);

NEG=zeros(size(data,1),size(data,3),2);%total number of copies for each group of POVM
for i=1:size(data,1)
    for j=1:size(data,3)
        NEG(i,j,1)=sum(data(i,:,j,1));
        NEG(i,j,2)=sum(data(i,:,j,2));
    end
end

epsx=zeros(size(data,1)/3,NAlpha,2);%the detection efficiency, as a comparison. And also depend on |alpha_n>
for i=1:size(epsx,1)
    for n=1:NAlpha
        epsx(i,n,1)=sum(sum(sum(data(3*(i-1)+1:3*i,1:4,n,1))))/sum(sum(sum(data(3*(i-1)+1:3*i,1:6,n,1))));
        epsx(i,n,2)=sum(sum(sum(data(3*(i-1)+1:3*i,1:4,n,2))))/sum(sum(sum(data(3*(i-1)+1:3*i,1:6,n,2))));
    end
end

aMax=3;%number of different outcomes for each POVM on Alice's side
xMax=size(data,1)/3;%number of different POVM on Alice's side
bMax=2;%number of different results for each POVM on Bob's side
yMax=3;%number of different POVM on Bob's side

E=zeros(2,2,bMax,yMax);%POVM on Bob/Charlie, 3rd index for outcome and 4th index for Bob's measurement direction
for j=1:yMax
    E(:,:,1,j)=(eye(2)+sigma(:,:,j))/2;
    E(:,:,2,j)=(eye(2)-sigma(:,:,j))/2;%corresponding to + and - measurements respectively
end

if yMax*xMax~=size(data,1)||bMax*aMax~=size(data,2)
    warndlg('Specified parameters are not consistent!','Warning');
end

MaxNy=max(max(max(sum(data,2))));% max number of measurement shots

epsilon=(sum(sum(sum(data(:,1:4,:,1))))+sum(sum(sum(data(:,1:4,:,2)))))/(sum(sum(sum(data(:,:,:,1))))+sum(sum(sum(data(:,:,:,2)))));%estimate the general detecting efficiency

% Assem=zeros(2,2,aMax,xMax,NAlpha,2);%Assemblage to be reconstructed, 3rd index for Alice outcome and 4th index for Alice measurement direction
% F=Assem;% assemblage=F'F
% T=zeros(2,2,NAlpha,2);
% rho0=eye(4)/4;epsx=ones(xMax,NAlpha,2)*epsilon;%the detection efficiency
% load('test2.mat');

gamma=zeros(xMax,NAlpha,2);%the portion to be transferred

%calculate the inital likelihood
LagStart=zeros(NAlpha,2);
for n=1:NAlpha
    for k=1:kMax
        for a=1:aMax
            for b=1:bMax
                for x=1:xMax
                    for y=1:yMax
                        LagStart(n,k)=LagStart(n,k)+data(yMax*(x-1)+y,bMax*(a-1)+b,n,k)*log(trace(Assem(:,:,a,x,n,k)*E(:,:,b,y)));
                    end
                end
            end
        end
    end
end
AssemStart=Assem;

tabL=zeros(51,n,k);
for n=1:NAlpha
    if UseWhich(n)~=0
        for k=1:kMax
            totaltimes=0;oldtotalL=LagStart(n,k);newtotalL=oldtotalL;
            while (newtotalL-oldtotalL>1||totaltimes<3)&&totaltimes<50
                oldtotalL=newtotalL;newtotalL=0;tabL(totaltimes+1,n,k)=oldtotalL;
            %testoldT=Assem(:,:,1,1,n,k)+Assem(:,:,2,1,n,k)+Assem(:,:,3,1,n,k);

                clear Sig T ;
                Sig{aMax,xMax}=sdpvar(2,2,'full','complex');
                for a=1:aMax
                    for x=1:xMax
                        Sig{a,x}=sdpvar(2,2,'full','complex');Sig{a,x}=(Sig{a,x}+Sig{a,x}')/2;
                    end
                end
                T=sdpvar(2,2,'full','complex');T = (T+T')/2;
    Fcons=[];

    %the following needs to change accordingly if you change aMax or xMax
                for x=1:xMax
                    Fcons=[Fcons,Sig{1,x}*(1-(gamma(x,n,k)>0)*gamma(x,n,k))+Sig{2,x}*(1+(gamma(x,n,k)<0)*gamma(x,n,k)) == T*epsx(x,n,k) ,Sig{1,x}+Sig{2,x}+Sig{3,x} == T,Sig{1,x}>=0,Sig{2,x}>=0,Sig{3,x}>=0];
                end
                Fcons=[Fcons,T>=0,trace(T)==1];
                obj=0;
                for x=1:xMax
                    for b=1:bMax
                        for y=1:yMax
                            obj=obj+data(yMax*(x-1)+y,bMax*(1-1)+b,n,k)*log(real(trace(Sig{1,x}*E(:,:,b,y)))) ...,
                                +data(yMax*(x-1)+y,bMax*(2-1)+b,n,k)*log(real(trace(Sig{2,x}*E(:,:,b,y)))) ...,
                                +data(yMax*(x-1)+y,bMax*(3-1)+b,n,k)*log(real(trace(Sig{3,x}*E(:,:,b,y))));
                        end
                    end
                end
                optimize(Fcons,-obj/10^5,ops);

                T=value(T);

                for a=1:aMax
                    for x=1:xMax
                        Assem(:,:,a,x,n,k)=value(Sig{a,x});
                    end
                end


                %try changing gamma_x
                for x=1:xMax
                    preLagx=0;
                    for b=1:bMax
                        for y=1:yMax
                            preLagx=preLagx+data(yMax*(x-1)+y,bMax*(1-1)+b,n,k)*log(real(trace(Assem(:,:,1,x,n,k)*E(:,:,b,y)))) ...,
                            +data(yMax*(x-1)+y,bMax*(2-1)+b,n,k)*log(real(trace(Assem(:,:,2,x,n,k)*E(:,:,b,y)))) ...,
                            +data(yMax*(x-1)+y,bMax*(3-1)+b,n,k)*log(real(trace(Assem(:,:,3,x,n,k)*E(:,:,b,y))));
                        end
                    end

                    timex=0;
                    stopsign=0;
                    difLag=0;
                    nowLagx=preLagx;
                    histLag=zeros(1,51);
                    histgamma=histLag;
                    stepcoe=1;

                    while timex<size(histLag,2)-1&&stopsign<2&&(difLag>0.01||timex<25)
                        lastLagx=nowLagx;
                        histLag(timex+1)=nowLagx;
                        histgamma(timex+1)=gamma(x,n,k);
                        oldgamma=gamma(x,n,k);
                        stepgamma=0.03/10^((totaltimes>0)+(totaltimes==2));%the least change magnitude of gamma_x
%10^(totaltimes>0)
                        %how about increase gamma a little
                        gamma(x,n,k)=oldgamma+(stopsign<1)*max(stepgamma,abs(difLag)/10*stepcoe)+(stopsign==1)*stepgamma/2;
                        clear Sig Fcons epsx0;
                        Sig{aMax}=sdpvar(2,2,'full','complex');
                        for a=1:aMax
                            Sig{a}=sdpvar(2,2,'full','complex');
                            Sig{a}=(Sig{a}+Sig{a}')/2;
                        end
                        epsx0=sdpvar(1);
            %the following needs to change accordingly if you change aMax or xMax
                        Fcons=[Sig{1}*(1-(gamma(x,n,k)>0)*gamma(x,n,k))+Sig{2}*(1+(gamma(x,n,k)<0)*gamma(x,n,k)) == T*epsx0 ,Sig{1}+Sig{2}+Sig{3} == T,Sig{1}>=0,Sig{2}>=0,Sig{3}>=0,epsx0>=0.5,epsx0<=0.8];
                        obj=0;
                        for b=1:bMax
                            for y=1:yMax
                                obj=obj+data(yMax*(x-1)+y,bMax*(1-1)+b,n,k)*log(real(trace(Sig{1}*E(:,:,b,y)))) ...,
                                +data(yMax*(x-1)+y,bMax*(2-1)+b,n,k)*log(real(trace(Sig{2}*E(:,:,b,y)))) ...,
                                +data(yMax*(x-1)+y,bMax*(3-1)+b,n,k)*log(real(trace(Sig{3}*E(:,:,b,y))));
                            end
                        end
                        optimize(Fcons,-obj/10^5,ops);
                        nowLag1=value(obj);nowgamma1=gamma(x,n,k);noweps1=value(epsx0);

                        %how about decrease gamma a little
                        gamma(x,n,k)=oldgamma-(stopsign<1)*max(stepgamma,abs(difLag)/10*stepcoe)-(stopsign==1)*stepgamma/2;
                        clear Sig Fcons epsx0;
                        Sig{aMax}=sdpvar(2,2,'full','complex');
                        for a=1:aMax
                            Sig{a}=sdpvar(2,2,'full','complex');Sig{a}=(Sig{a}+Sig{a}')/2;
                        end
                        epsx0=sdpvar(1);
            %the following needs to change accordingly if you change aMax or xMax
                        Fcons=[Sig{1}*(1-(gamma(x,n,k)>0)*gamma(x,n,k))+Sig{2}*(1+(gamma(x,n,k)<0)*gamma(x,n,k)) == T*epsx0 ,Sig{1}+Sig{2}+Sig{3} == T,Sig{1}>=0,Sig{2}>=0,Sig{3}>=0,epsx0>=0.5,epsx0<=0.8];
                        obj=0;
                        for b=1:bMax
                            for y=1:yMax
                                obj=obj+data(yMax*(x-1)+y,bMax*(1-1)+b,n,k)*log(real(trace(Sig{1}*E(:,:,b,y)))) ...,
                                +data(yMax*(x-1)+y,bMax*(2-1)+b,n,k)*log(real(trace(Sig{2}*E(:,:,b,y)))) ...,
                                +data(yMax*(x-1)+y,bMax*(3-1)+b,n,k)*log(real(trace(Sig{3}*E(:,:,b,y))));
                            end
                        end
                        optimize(Fcons,-obj/10^5,ops);
                        nowLag2=value(obj);
                        nowgamma2=gamma(x,n,k);
                        noweps2=value(epsx0);

                        if nowLag1<lastLagx&&nowLag2<lastLagx
                            stopsign=stopsign+1;
                            gamma(x,n,k)=oldgamma;
                            nowLagx=lastLagx;
                        elseif nowLag1>lastLagx&&nowLag2>lastLagx
                            stopsign=0;
                            gamma(x,n,k)=(nowLag1>=nowLag2)*nowgamma1+(nowLag1<nowLag2)*nowgamma2;
                            nowLagx=max(nowLag1,nowLag2);
                            difLag=nowLagx-lastLagx;
                            epsx(x,n,k)=(nowLag1>=nowLag2)*noweps1+(nowLag1<nowLag2)*noweps2;
                        else
                            stopsign=0;
                            gamma(x,n,k)=(nowLag1>=nowLag2)*nowgamma1+(nowLag1<nowLag2)*nowgamma2;
                            nowLagx=max(nowLag1,nowLag2);
                            difLag=nowLagx-lastLagx;
                            epsx(x,n,k)=(nowLag1>=nowLag2)*noweps1+(nowLag1<nowLag2)*noweps2;
                        end
                        timex=timex+1;
                    end


                    newtotalL=newtotalL+nowLagx;
                end
                totaltimes=totaltimes+1;
            end
        end
    end
end

%plot(1:timex,histLag(1:timex));figure;plot(1:timex,histgamma(1:timex));

Tfinal=zeros(2,2,NAlpha,2);%the final value of rho_B
for n=1:NAlpha
    if UseWhich(n)~=0
        for k=1:kMax
            clear Sig T ;
            Sig{aMax,xMax}=sdpvar(2,2,'full','complex');
            for a=1:aMax
                for x=1:xMax
                    Sig{a,x}=sdpvar(2,2,'full','complex');
                    Sig{a,x}=(Sig{a,x}+Sig{a,x}')/2;
                end
            end
            T=sdpvar(2,2,'full','complex');
            T = (T+T')/2;
            Fcons=[];

            %the following needs to change accordingly if you change aMax or xMax
            for x=1:xMax
                Fcons=[Fcons,Sig{1,x}*(1-(gamma(x,n,k)>0)*gamma(x,n,k))+Sig{2,x}*(1+(gamma(x,n,k)<0)*gamma(x,n,k)) == T*epsx(x,n,k) ,Sig{1,x}+Sig{2,x}+Sig{3,x} == T,Sig{1,x}>=0,Sig{2,x}>=0,Sig{3,x}>=0];
            end
            Fcons=[Fcons, T>=0,trace(T)==1];
            obj=0;
            for x=1:xMax
                for b=1:bMax
                    for y=1:yMax
                        obj=obj+data(yMax*(x-1)+y,bMax*(1-1)+b,n,k)*log(real(trace(Sig{1,x}*E(:,:,b,y)))) ...,
                            +data(yMax*(x-1)+y,bMax*(2-1)+b,n,k)*log(real(trace(Sig{2,x}*E(:,:,b,y)))) ...,
                            +data(yMax*(x-1)+y,bMax*(3-1)+b,n,k)*log(real(trace(Sig{3,x}*E(:,:,b,y))));
                    end
                end
            end

            optimize(Fcons,-obj/10^5,ops);

            T=value(T);
            Tfinal(:,:,n,k)=value(T);
            for x=1:xMax%newly added to increase numerical accuracy
                clear Sig Fcons epsx0;
                Sig{aMax}=sdpvar(2,2,'full','complex');
                for a=1:aMax
                    Sig{a}=sdpvar(2,2,'full','complex');
                    Sig{a}=(Sig{a}+Sig{a}')/2;
                end
                epsx0=sdpvar(1);
            %the following needs to change accordingly if you change aMax or xMax
                Fcons=[Sig{1}*(1-(gamma(x,n,k)>0)*gamma(x,n,k))+Sig{2}*(1+(gamma(x,n,k)<0)*gamma(x,n,k)) == T*epsx0 ,Sig{1}+Sig{2}+Sig{3} == T,Sig{1}>=0,Sig{2}>=0,Sig{3}>=0,epsx0>=0.5,epsx0<=0.8];
                obj=0;
                for b=1:bMax
                    for y=1:yMax
                        obj=obj+data(yMax*(x-1)+y,bMax*(1-1)+b,n,k)*log(real(trace(Sig{1}*E(:,:,b,y)))) ...,
                        +data(yMax*(x-1)+y,bMax*(2-1)+b,n,k)*log(real(trace(Sig{2}*E(:,:,b,y)))) ...,
                        +data(yMax*(x-1)+y,bMax*(3-1)+b,n,k)*log(real(trace(Sig{3}*E(:,:,b,y))));
                    end
                end
                optimize(Fcons,-obj/10^5,ops);
                epsx(x,n,k)=value(epsx0);
                for a=1:aMax
                    Assem(:,:,a,x,n,k)=value(Sig{a});
                end

            end
        end
    end
end

%calculate the corresponding likelihood
Lag=zeros(NAlpha,2);
for n=1:NAlpha
    if UseWhich(n)~=0
    for k=1:kMax
        for b=1:bMax
            for x=1:xMax
                for y=1:yMax
                    Lag(n,k)=Lag(n,k)+data(yMax*(x-1)+y,bMax*(1-1)+b,n,k)*log(real(trace(Assem(:,:,1,x,n,k)*E(:,:,b,y)))) ...,
                            +data(yMax*(x-1)+y,bMax*(2-1)+b,n,k)*log(real(trace(Assem(:,:,2,x,n,k)*E(:,:,b,y)))) ...,
                            +data(yMax*(x-1)+y,bMax*(3-1)+b,n,k)*log(real(trace(Assem(:,:,3,x,n,k)*E(:,:,b,y))));
                end
            end
        end
    end
    end
end

%plot(1:size(threshhistory,2),threshhistory);
%threshhistory(times)
teste1=zeros(xMax,NAlpha,2);
errMax=0;%to what accuracy are the conditions are satisfied
nonsignal=zeros(2,2,xMax,NAlpha,2);%the matrix to verify whether non-signaling condition is satisfied
for n=1:NAlpha
    if UseWhich(n)~=0
        for k=1:kMax
            for x=1:xMax
                for a=1:aMax
                    nonsignal(:,:,x,n,k)=nonsignal(:,:,x,n,k)+Assem(:,:,a,x,n,k);
                    [vs,ds]=eig(Assem(:,:,a,x,n,k));
                    if sum(diag(ds)<0)>0
                        errMax=1;%check if every assemblage is positive semidefinite
                    end
                end
                errMax=max(errMax,norm(nonsignal(:,:,1,n,k)-nonsignal(:,:,x,n,k),'fro'));%check if nonsignaling condition is satisfied
                %errMax=max(errMax,norm(nonsignal(:,:,1,n,k)/trace(nonsignal(:,:,1,n,k))-Assem(:,:,aMax,x,n,k)/trace(Assem(:,:,aMax,x,n,k)),'fro'));%check if
                %null assemblage is proportional to Bob's state
                teste1(x,n,k)=norm(nonsignal(:,:,1,n,k)*epsx(x,n,k)-Assem(:,:,1,x,n,k)*(1-(gamma(x,n,k)>0)*gamma(x,n,k))-Assem(:,:,2,x,n,k)*(1+(gamma(x,n,k)<0)*gamma(x,n,k)),'fro');
                if errMax>0.01
                    zzz=1;
                end
            end
            errMax=max(errMax,abs(1-trace(nonsignal(:,:,1,n,k))));
            if errMax>0.01
                zzz=1;
            end
        end
    end
end


idealp=0*data;%the ideal probabilities (would be) generated by our estimation without any noise
for n=1:NAlpha
    if UseWhich(n)~=0
        for k=1:kMax
            for y=1:yMax
                for b=1:bMax
                    for x=1:xMax
                        for a=1:aMax
                            idealp(yMax*(x-1)+y,bMax*(a-1)+b,n,k)=real(trace(Assem(:,:,a,x,n,k)*E(:,:,b,y)));
                        end
                    end
                end
            end
        end
    end
end

expf=0*data;%experiment frequency
MultinomialTest=zeros(size(data,1),NAlpha,2);
for povmn=1:size(data,1)
    for alphan=1:NAlpha
        if UseWhich(alphan)~=0
            for k=1:kMax
                pmle0=idealp(povmn,:,alphan,k);
                ndata1=data(povmn,:,alphan,k);ndata=sum(ndata1);tdata=ndata1/ndata;expf(povmn,:,alphan,k)=tdata;
                MultinomialTest(povmn,alphan,k)=-2*sum(ndata1.*log(pmle0./(tdata+(tdata==0))))/(1+(sum(1./pmle0)-1)/6/ndata/(size(data,2)-1));
            %Previous: MultinomialTest(povmn,alphan,k)=-2*sum(ndata1.*log(pmle0./tdata))/(1+(sum(1./ndata1)-1)/6/ndata/(size(data,2)-1));
   %the divided factor is for amendment, see Wiki: multinomial test;
   %'+(tdata==0)' is because the zero value's contribution is zero
            end
        end
    end
end
MTestResult=sum(sum(sum(MultinomialTest)));  %it should follow a chi^2 distribution with df degrees of freedom
df=round(size(data,1)*(size(data,2)-1))*sum(UseWhich~=0)*kMax;%size(data,4)

temv1=(data>-1);
for n=1:NAlpha
    if UseWhich(n)==0
        temv1(:,:,n,:)=0;
    end
end
temv2=(idealp-expf).*temv1;temv3=expf.*temv1;
RelativeFreqErr=(norm(temv2(:),'fro'))/(norm(temv3(:),'fro'));
% disp('[errMax,RelativeFreqErr,MTest,DFreedom]')
% disp([errMax,RelativeFreqErr,MTestResult,df])

gamma=-gamma;%in the old version gamma was on the LHS of the non-signalling equation for the null results. Now it is on the RHS, so reversed
clear epsx0 Fcons obj ops Sig
end
% save exp2_Problem_5_faster;
