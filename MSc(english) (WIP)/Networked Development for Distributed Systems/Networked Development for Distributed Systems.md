
[https://docs.google.com/spreadsheets/d/1HPtbfsIuI4NVwkv5OaBgSClLjLNcoMWhH8nX7A58e9U/edit#gid=1577429949](https://docs.google.com/spreadsheets/d/1HPtbfsIuI4NVwkv5OaBgSClLjLNcoMWhH8nX7A58e9U/edit#gid=1577429949)


- Event based communication: [00.Apache Kafka](src/00.Apache%20Kafka.md)
- Actor model: [01.Akka](src/01.Akka.md) 
- Data analysis: Apache Spark
- Edge Computing: Node-red
- High performance computing: MPI
- IoT: Contiki

The course argues that any modern engineer should be proficient in writing networked software due to its diversity and applicability across different platforms and technologies.




- **Publisher/Subscriber Paradigm:**
    
    - Messages sent to topics, not specific receivers.
    - Ideal for broadcasting to multiple subscribers.
    - Publishers unaware of subscribers.
- **Producer/Consumer Paradigm:**
    
    - Producers send data to a specific queue.
    - Consumers retrieve and process data from the queue.
    - Suited for task distribution and processing.
    - Focus on load balancing and message processing scaling.
