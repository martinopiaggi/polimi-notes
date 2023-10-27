import Control.Monad.State.Lazy
import Data.List (isInfixOf)

lun :: [a] -> Integer
lun [] = 0
lun (x:xs) = 1 + lun xs

fuck :: Integer -> Integer
fuck 0 = 1
fuck n = n*fuck(n-1)

rev :: [a] -> [a]
rev [] = []
rev (x:xs) = rev (xs) ++ [x]

data Human = Woman | Man

instance Show Human where 
    show Woman = "donnaa"
    show Man = "uomooo"

instance Eq Human where
    Man == Man = True
    Woman == Woman = True
    _ == _ = False


data Point = Point Float Float deriving (Show,Eq)

getx (Point a _) = a
gety (Point _ a) = a

euler :: Point -> Point -> Float
euler (Point x1 y1) (Point x2 y2) = 
    let dx = x1 - x2 
        dy = y1 - y2 
    in sqrt ( dx*dx + dy*dy )


zippa :: [a] -> [b] -> [(a,b)]
zippa [] _ = []
zippa _ [] = []
zippa (x:xs) (y:ys) = (x,y) : zippa (xs) (ys)

isEven :: Integer -> Bool
isEven x = ((mod x 2) == 0)

filtra :: (a -> Bool) -> [a] -> [a]
filtra _ [] = []
filtra x (u:s) 
    | x u = u : filtra x s
    | otherwise = filtra x s 

-- foldable stuff

somma xs = foldl (\acc x -> acc + x) 0 xs

isThere y ys = foldl (\acc x -> if x == y then True else acc) False ys

fucktransform xs = [(++"fuck"),(++"test")] <*> xs

foo x = do  
    let y = x -1
    show y
    let x = x + 1 
    show (x - y)

bmi :: Float -> Float -> String 
bmi w h 
    | calc <= 18.5 = "Underweight"
    | calc <= 25.0 ="Normal"
    | calc <= 30.0 = "Overweight"
    | otherwise = "Obese"
    where calc = w/h^2

--SORTING

find :: Int -> [Int] -> Int 
find elem array 
    | head array == elem = 0
    | otherwise = 1 + (find elem (tail array))

replace :: Int -> Int -> [Int] -> [Int]
replace elem pos array 
    | pos == 0 = [elem] ++ (tail array)
    | otherwise = [head array] ++ (replace elem (pos-1) (tail array)) 

swapByValue :: Int -> Int -> [Int] -> [Int]
swapByValue x y array = do 
    let posX = (find x array)
    let posY = (find y array)
    replace y posX (replace x posY array) 
    
swap :: Int -> Int -> [Int] -> [Int]
swap posX posY array = do 
    let x = (array!!posX)
    let y = (array!!posY)
    replace y posX (replace x posY array) 

-- remember that head/last turns a single element, not a list of a single element

selectionSort x
    | length x == 0 = []
    | otherwise = do 
        let y = swapByValue (head x) (minimum x) x
        [head y] ++ (selectionSort (tail y))

-- not in place Quicksort version:

middle array= array!!((length array) `div` 2)
lower array = filter ( < middle array) array
greater array = filter ( > middle array) array

fakeQuicksort array 
    | length array == 0 = []
    | otherwise = (fakeQuicksort (lower array)) 
    ++ [middle array]
    ++ (fakeQuicksort (greater array))


-- QUICKSORT IN PLACE, probably not "enough functional"

splitr pos array = 
    if pos>=0  
        then splitr (pos-1) (tail array) 
        else array

splitl pos array = splitlHelper ((length array) -1 -pos) array

splitlHelper pos array = 
    if pos>=0  
        then splitlHelper (pos-1) (init array) 
        else array

pivoting i j (array, pivotPos) 
    | length array <= 1 = (array, pivotPos)
    | i>j = ((swap j pivotPos array), j)
    | x <= pivot  = pivoting (i+1) j (array, pivotPos)
    | y > pivot = pivoting i (j-1) (array, pivotPos)
    | pivotPos == i = pivoting (i+1) (j-1) ((swap i j array), j) 
    | pivotPos == j = pivoting (i+1) (j-1) ((swap i j array), i) 
    | otherwise = pivoting (i+1) (j-1) ((swap i j array), pivotPos) 
    where 
        x = array!!i
        y = array!!j
        pivot = array!!pivotPos

quicksort array = do 
    let (pivoted,pivotPos) = pivoting 0 ((length array)-1) (array,((length array)`div`2))
    if (length pivoted <= 2) 
        then pivoted
        else do 
            let left = splitl pivotPos pivoted
            let right = splitr pivotPos pivoted
            (quicksort left) ++ [pivoted!!pivotPos] ++ (quicksort right)
            
--exercises of 18/11 

-- define data Listree


-- data Listree a = Cons a (Listree a) | Null | Branch (Listree a) (Listree a) deriving (Show,Eq)

--instance Functor Listree where 
--    fmap f Null = Null 
--    fmap f (Cons x t) = Cons (f x) (fmap f t)
--    fmap f (Branch t1 t2) = Branch (fmap f t1) (fmap f t2)

--instance Foldable Listree where 
--    foldr f z Null = z 
--    foldr f z (Cons x t) = f x (foldr f z t)
--    foldr f z (Branch t1 t2) = foldr f (foldr f z t2) t1 --look very well this


-- STUDING FROM START AGAIN

square :: Integer -> Integer
square x = x*x

ev :: Integer -> Bool 
ev x  
    | x `mod` 2  == 0 = True
    | otherwise = False


fact :: Integer -> Integer  
fact 0 = 1 
fact n = n * fact (n-1)


lunghezza :: [a] -> Int
lunghezza [] = 0
lunghezza x = 1 + lunghezza (tail x)

myReverse :: [a] -> [a]
myReverse (x : xs) = myReverse xs ++ [x] 
myReverse [] = [] 


mappazza f l
    | l == [] = []
    | otherwise = [f (head l)] ++ (mappazza f (tail l))


-- binary function 
-- plus(x,y) {return x + y;}  
plus :: Num a => a -> a
plus = (\x->x*x)

showAPiece :: Show a => [a] -> [Char]
showAPiece [] = ""
showAPiece (x:xs) = show x ++ "|" ++ show xs 

foldaL :: (t1 -> t2 -> t2) -> t2 -> [t1] -> t2 
foldaL f acc [] = acc 
foldaL f acc (x:xs) = foldaL f (f x acc) xs

----

data Tree a = Empty | Leaf a | Node (Tree a) (Tree a)

instance Show x => Show (Tree x) where
    show Empty = "()"
    show (Leaf a) = "(" ++ show a ++ ")"
    show (Node a b) = "[" ++ show a ++ show b ++ "]"

foldT :: (b -> a -> b) -> b -> (Tree a) -> b 
foldT _ acc Empty = acc 
foldT f acc (Leaf x) = f acc x
foldT f acc (Node l r) = (foldT f (foldT f acc l) r)

instance Foldable Tree where
    foldl = foldT

ftmap :: (a->b) -> (Tree a) -> (Tree b)
ftmap _ Empty = Empty 
ftmap f (Leaf x) = (Leaf (f x))
ftmap f (Node z1 z2) = (Node (ftmap f z1) (ftmap f z2)) 

instance Functor Tree where 
    fmap = ftmap

instance Applicative Tree where 
    pure x = (Leaf x)
    _ <*> Empty = Empty
    Leaf f1 <*> Leaf x = Leaf (f1 x)
    Node fl fr <*> Node l r = 
        Node (Node (fl <*> l) (fl <*> r))
        (Node (fr <*> l) (fr <*> r))

instance Monad Tree where
  return = pure
  Empty >>= f = Empty
  Leaf x >>= f = f x
  Node l r >>= f = Node (l >>= f) (r >>= f)


--- FUNCTOR APPLICATIVE and MONADS with a logger 

type Log = [String] --remember that is just an alias
data Logger a = Logger a Log

instance (Eq a) => Eq (Logger a) where
    (Logger x _) == (Logger y _) = x == y

instance (Show a) => Show (Logger a) where
   show (Logger x s) = show x ++ "{" ++ show s ++ "}"

instance Functor Logger where 
    fmap f (Logger x s) = Logger (f x) s 

instance Applicative Logger where
    pure x = Logger x ["Init"]
    Logger f f_name <*> (Logger x log) =  (Logger (f x) (log ++ f_name))

instance Monad Logger where 
    return = pure 
    (Logger x l) >>= f = 
        let Logger x' l' = f x
        in Logger x' (l ++ l') 

plusLog num x = Logger (x+num) [("Added " ++ show num)]

--- 

makeStream :: [l] -> [l]
makeStream l = l ++ makeStream l

--- 

data Btree a =  Btree (a,Int,Btree a, Btree a) | Nil

instance (Show o) => Show (Btree o) where
    show Nil = ""
    show (Btree (a,x,l,r)) = "{"++ show a ++ "(" ++ show x ++ ")" ++ show l ++ show r ++ "}"

addLeft :: Btree a -> Btree a -> Btree a
addLeft(Btree (a1,num1,Nil, r1)) new@(Btree (a2,num2,l2, r2)) =
    (Btree (a1,num1+1+num2,new,r1)) 
addLeft(Btree (a1,num1,Btree (_,oldNum,_,_), r1)) new@(Btree (a2,num2,l2, r2)) =
    (Btree (a1,num1+1+num2-oldNum,new,r1)) 

instance Functor Btree where
    fmap f (Btree (a,num,l,r)) = (Btree (f a,num,fmap f l,fmap f r))
    fmap f Nil = Nil

instance Foldable Btree where
    foldl f acc Nil = acc
    foldl f acc (Btree (a,num,l,r)) = f (foldl f (foldl f acc l) r) a
    

--- Stack random --- 

--- Immutable stack --- 

type Stack = [Int]

pop :: Stack -> (Stack, Int)
pop [] = error "Pop on empty stack"
pop (x:xs) = (xs,x) 

push :: Stack -> Int -> Stack 
push x p = x ++ [p]

-- Mutable stack  

-- data State st a = State (st -> (st,a))
-- there is always an implicit and an explicit part of the monad 
-- monadic code use do notation, so it's ysy 

-- monadic action

popM :: State Stack Int 
popM = do 
    stack <- get  -- get is used to " get the state"
    case stack of 
        (x:xs) -> put xs >> return x
        [] -> error "Pop on empty stack"

pushM :: Int -> State Stack () 
pushM x = do 
    stack <- get  -- get is used to " get the state"
    put (x:stack) -- put is used to " set the state"

-- in monadic action everything is like imperative code
-- then you have to use "RunState"

monadicActionOnStack = do 
    pushM 2
    pushM 1
    popM
    pushM 4



-- Exam 2020/01/15

-- make it an instance of functor,applicative and monad 


data CashReg a = CashReg { getReceipt :: (a,Float) } deriving (Show, Eq)

getCurrentItem = fst . getReceipt 
getPrice = snd . getReceipt

instance Functor CashReg where 
    fmap f cr = CashReg (f $ getCurrentItem cr, getPrice cr)

instance Applicative CashReg where 
    pure a = CashReg (a,0.0)
    fcr <*> cr = CashReg (getCurrentItem fcr $ getCurrentItem cr, getPrice fcr + getPrice cr) 

instance Monad CashReg where 
    return = pure 
    cr >>= f = 
        let newCr = f $ getCurrentItem cr 
        in CashReg (getCurrentItem newCr, getPrice cr + getPrice newCr)


addItem :: String -> Float -> CashReg String 
addItem item price = CashReg (item,price)


checkBag :: String -> CashReg String 
checkBag item = if isInfixOf "veg" item
                    then CashReg ("bag", 0.01)
                    else CashReg ("",0.0)

buyStuff = do 
    i1 <- addItem "veg pro" 10.0 -- When you use `<-`, it unwraps the value from its monadic context
    checkBag i1
    i2 <- addItem "milk" 1.0
    checkBag i2


data Queue a = End | Cons a (Queue a)

instance Show q => Show (Queue q) where
    show End = ""
    show (Cons x y) = (show y) ++ " " ++ (show x)


gzip xs = if null (filter null xs) 
          then (map head xs) : gzip (map tail xs) 
          else []


powset set = helper set [[]] where
helper [] out = out
helper ( e : set ) out = helper set ( out ++ [e : x | x <- out ])
