L = load('C:/Users/Administrator/Desktop/毕业论文/答辩报告/L.txt');
IL = transpose(sum(L));
IB = sum(L,2);
N = size(L,1);
p = 0.8;
Aeq = ones(N,N);
beq = sum(IL / p) * ones(N,1);
lb = max(IB,IL) * 1.01;
global IL,p,N;
x0 = ones(N,1) * sum(IL) / p / N;
[BA,funvalue] = fmincon(@Heter,x0,[],[],Aeq,beq,lb);
save C:/Users/Administrator/Desktop/毕业论文/答辩报告/BA.txt -ascii BA
