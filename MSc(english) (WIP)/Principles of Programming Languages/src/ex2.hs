
data Queue a = Queue [a] [a] 

instance (Show a) => Show (Queue a) where 
  show (Queue x y) = show x ++ "|" ++ show y  

instance (Eq a) => Eq (Queue a) where
    (Queue x y) == (Queue x1 y1) = (x == x1 && y == y1)

to_list (Queue x y) = x ++ reverse y

enqueue :: a -> Queue a -> Queue a
enqueue b (Queue x y) = Queue x (b:y)  

-- prof did it with also "Maybe" :

dequeue :: Queue a -> Queue a
dequeue (Queue (x:xs) y) = Queue (xs) y 
dequeue (Queue [] y ) = dequeue (Queue (reverse y) [])

instance Functor Queue where 
    fmap f (Queue x y) = Queue (fmap f x) (fmap f y)


-- professors did it differently 
instance Applicative Queue where
    pure x = Queue [] [x]
    Queue f1 f2 <*> (Queue x y) = Queue (f1 <*> x) (f2 <*> y)


-- 2020 02 07

data Pricelist a = Pricelist [(a,Float)] deriving (Show,Eq)


pmap :: (a->b) -> Float -> Pricelist a -> Pricelist b
pmap f v (Pricelist xs) = 
        Pricelist 
        (fmap (\x -> let (a,p) = x in (f a, p + v)) xs)

instance Functor Pricelist where 
    fmap f x = pmap f 0.0 x

foldrPL :: (a -> b -> b) -> b -> (Pricelist a) -> b  
foldrPL _ acc (Pricelist []) = acc
foldrPL f acc (Pricelist (x:xs)) = 
     foldrPL f (f (fst x) acc) (Pricelist xs)


data Tril a = Tril [a] [a] [a] deriving (Show,Eq)

instance Functor Tril where 
    fmap f (Tril x y z) =
        Tril (fmap f x) (fmap f y) (fmap f z)

instance Foldable Tril where 
    foldr f acc (Tril x y z) =
        foldr f (foldr f (foldr f acc z) y) x

(Tril x y z) +++ (Tril a b c) = 
    Tril (x ++ y) (z ++ a) (b ++ c)

