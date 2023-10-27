-module(exe).
-export([add/2,car/1,cdr/1,map/2,
         execute/3,execute2/3,pmap/2,pfilter/2,filterr/2,
         printFrom/1,setAlarm/2,set/3,proxyFun/1,
         rangee/3,next/1,iterator/4]).

add(A,B) -> A + B.
car([X|_]) -> X.
cdr([X|Xs]) -> Xs.

map(_, []) -> [];
map(F,[X|Xs]) -> [F(X) | map(F,Xs)].

filterr(F, [H|T]) ->
    case F(H) of
        true  -> [H|filterr(F, T)];
        false -> filterr(F, T)
    end;
filterr(F, []) -> [].

%parallel versions


% parallel map
pmap(F, L) ->
    Ps = [ spawn(?MODULE, execute, [F, X, self()]) || X <- L],
    [receive 
         {Pid, X} -> X
     end || Pid <- Ps].

execute(F, X, Pid) ->
    Pid ! {self(), F(X)}.



%same as pmap but we have to discard
pfilter(F, L) ->
    Ps = [ spawn(?MODULE, execute2, [F, X, self()]) || X <- L],
    lists:foldl(fun (P,Vo) ->
               receive
                  {P,true, X} -> Vo ++ [X];
                  {P,false,_} -> Vo
                end end, [], Ps).

execute2(F, X, Pid) ->
    case F(X) of
        true  -> Pid ! {self(), true,X};
        false -> Pid ! {self(), false,X}
    end.

printFrom(0) -> 
    [];
printFrom(N) ->
    [N] ++ printFrom(N-1). 


setAlarm(T,What)->
    spawn(?MODULE,set,[self(),T,What]),
    receive
        {Alarm} -> io:format("~s",[Alarm])
    end.

set(Pid,T,Alarm) ->
    receive
    after 
        T -> Pid ! {Alarm}
    end.


%2021 01 20 
%proxy 

proxyFun(Table) -> 
    receive 
        {remember, Pid, Name} -> 
            proxyFun(Table#{Name => Pid});
        {question, Name, Data} ->
            #{Name := Id} = Table,
            Id ! {question,Data},
            proxyFun(Table);
        {answer, Name, Data} ->
            #{Name := Id} = Table,
            Id ! {answer, Data},
            proxyFun(Table)
    end. 


% 2020 01 15

rangee(Startv,Endv,Step)->spawn(?MODULE,iterator,[self(),Startv-Step,Endv,Step]).

next(R) -> 
    R ! {self(),next},
    receive
        {Value} -> Value;
        stop_iteration -> stop_iteration
    end.

iterator(Pid,Startv,Endv,Step) ->
    receive {Pid,next} -> 
        case (Startv+Step)<Endv of
            true  -> Pid ! {Startv + Step}, iterator(Pid,Startv + Step, Endv, Step);
            false -> Pid ! stop_iteration
        end
    end.


