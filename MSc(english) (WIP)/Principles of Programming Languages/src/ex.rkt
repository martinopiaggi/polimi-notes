

; esempio interessante

(define (minimum L) 
(let ((prev (car L))
  (nex (cdr L)) )
  (if (null? nex)

prev

(if (< (car nex) prev)

(minimum (cons (car nex) (cdr nex) )) 

(minimum (cons prev (cdr nex) )) 

)

)
)
)

;(minimum '(42 43 52 59 50 60) )




;prima esercitazione 




; lunghezza lista


(define (len L)

  (if (null? L)

      0
      
      (+ 1 (len (cdr L)))) )

; (len '(21 2 5 1 1 1 1 1))


; trovare prefisso prime n cifre


(define (prefix n L)
(define (inner n L F)
  (if (= n 0)
      F
      (inner (- n 1) (cdr L) (cons F (car L)))))

  (inner (- n 1) (cdr L) (car L))
  )

;(display (prefix 4 '(10 1 2 3 4 9 12)))

; wrong ... bad formatting

(define (prefiy n L)
  (if (= n 0)
      '()
      (cons (car L) (prefiy (- n 1) (cdr L))) 
  )
)

;(display (prefiy 4 '(10 1 2 3 4 9 12)))


;range def



; reverse list


;(display (revers  '(1 2 3 4 9)))
                                    
; flat function <- In this one i forgot the 'list; conversion !! I saw the solutions..

(define (flat L)
  (if (null? L)
      '()
      (append (if (list? (car L)) (flat (car L)) (list (car L))) (flat (cdr L))) ))

;(flat '( 3 (1 2) 2))



; Third lesson random stuff 

(struct human (
               name
               age ))


(define Martino (human "Martino" 22))


(define (say-hello x)
  (if (human? x)
      (begin (display "Hi! I'm ")
      (display (human-name x))
      (newline))

      (display "I'm not human")
      ))

;(say-hello Martino)
;(say-hello 4)


;iterator using a closure!


(define (iterator vec)
  (let ((cur 0)
    (top (vector-length vec)))
    (lambda ()
      (if (= cur top)
          '<<end>>
          (let ((v (vector-ref vec cur)))
            (set! cur (+ cur 1))
            (display v)
            (newline)
          )))))




(define i (iterator #(1 2 3 4 5)))

;(i)
;(i)
;(i)



;basically the closures are obtained using set! (bang procedure!)


(define-syntax while
  (syntax-rules ()
    ((_ condition body ...)
     (let loop ()
       (when condition
         (begin body ... (loop)))))
    ))

(define (test x)
  (while (< x 3) (display x) (set!  x (+ x 1))))

;(test 0)



; del 12-10 binary tree ( OO)

(struct node-base
  ((value #:mutable)))

(struct node node-base ;per indicare inheritance
  (left right))


  
;fibonacci

(define lastFibonacci 1)
(define lastLastFibonacci 0)

(define (fibonacciNext) 
  (let ((swap lastFibonacci))
    (set! lastFibonacci (+ lastFibonacci lastLastFibonacci))
    (set! lastLastFibonacci swap)
    (display lastFibonacci)
    (newline)
  )
  )



; standard usage of continuations

(define (break-negative l)
  (call/cc (lambda (break) ;using continuations as 'escape function'
            (for-each (lambda (x)
                        (if (< x 0)
                            (break)
                            (displayln x)))
                       l))))

;(break-negative '(40 41 42 -43 44))


(define (continue-negative l)
  (for-each (lambda (x)
              (call/cc (lambda (continue)
                         (if (< x 0) (continue) (displayln x)
                             )))) l ))
              
                
;(continue-negative '(40 41 42 -43 44))

;2021 01 20
          
(define (mlv l)
  (if (list? l)
  (apply vector (map mlv l))
  l
  ))

; 2020 07 17 



; closures as classes

(define (make-person name age)
  (define (get-name) name)
  (define (get-age) age)

  (lambda (message . args)
    (case message 
      ((get-name) get-name)
      ((get-age) get-age)
    )))


(define (f . L) 

(foldl (lambda (y1 y2) (cons (list y1) y2)) (foldl cons '() L) L)

)

(define (mix f . rest )
  (foldr (lambda (x acc) (list x acc x)) 
  (map f rest)
  rest
  ))



(define (test-closures)
  (define (greet)
    (display "Hello"))
  
  (define (farewell)
    (display "Goodbye"))

  (lambda (message)
    ((case message
       ((greet) greet)
       ((farewell) farewell)
      ))))





















