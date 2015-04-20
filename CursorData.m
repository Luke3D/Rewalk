clear T1 T2
lci = length(cursor_info)
t1 = lci:-2:2
t2 = lci-1:-2:1

for ci = 1:length(t1)
    T1(ci) = cursor_info(t1(ci)).DataIndex;
    T2(ci) = cursor_info(t2(ci)).DataIndex;
end
T1 = T1';
T2=T2';

T1 = sort(T1);
T2 = sort(T2);
