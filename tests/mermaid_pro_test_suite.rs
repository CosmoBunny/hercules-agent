use ratatui::text::Line;
use hercules_agent::diagram::DiagramRenderer;

fn lines_to_text(lines: &[Line]) -> String {
    lines
        .iter()
        .map(|l| {
            l.spans.iter().map(|s| s.content.as_ref()).collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// =========================================================================
// LEVEL 1: GROUND LEVEL BASICS (Node Shapes, Edge Types, Orientations)
// =========================================================================

#[test]
fn test_ground_01_basic_flowchart_shapes() {
    let body = r#"
    graph TD
        A[Square Box] --> B(Rounded Box)
        B --> C([Stadium Shape])
        C --> D[[Subroutine Box]]
        D --> E[(Database Cylinder)]
        E --> F((Circle Node))
        F --> G{Rhombus Decision}
        G --> H{{Hexagon Node}}
    "#;
    let lines = DiagramRenderer::render_to_lines("mermaid", body, 120);
    assert!(!lines.is_empty(), "Rendered output should not be empty");
    let text = lines_to_text(&lines);
    assert!(text.contains("Square Box") || text.contains("Square"));
    assert!(text.contains("Database Cylinder") || text.contains("Database"));
    assert!(text.contains("Rhombus Decision") || text.contains("Decision"));
}

#[test]
fn test_ground_02_all_edge_types_and_labels() {
    let body = r#"
    graph LR
        A[Alpha] --- B[Bravo]
        B -->|Label 1| C[Charlie]
        C -.-> D[Delta]
        D ==>|Thick Arrow| E[Echo]
        E --o F[Foxtrot]
        F --x G[Golf]
    "#;
    let lines = DiagramRenderer::render_to_lines("mermaid", body, 120);
    assert!(!lines.is_empty());
    let text = lines_to_text(&lines);
    assert!(text.contains("Alpha"));
    assert!(text.contains("Bravo"));
    assert!(text.contains("Charlie"));
    assert!(text.contains("Delta"));
    assert!(text.contains("Echo"));
}

#[test]
fn test_ground_03_all_directions() {
    for dir in &["TD", "TB", "LR", "RL", "BT"] {
        let body = format!("graph {}\n    Node1[Start Point] --> Node2[End Point]", dir);
        let lines = DiagramRenderer::render_to_lines("mermaid", &body, 100);
        assert!(!lines.is_empty());
        let text = lines_to_text(&lines);
        assert!(text.contains("Start Point") || text.contains("Start"));
        assert!(text.contains("End Point") || text.contains("End"));
    }
}

// =========================================================================
// LEVEL 2: INTERMEDIATE DIAGRAM TYPES (Sequence, Class, State, ER, Gantt, Pie, Git)
// =========================================================================

#[test]
fn test_intermediate_01_sequence_diagram() {
    let body = r#"
    sequenceDiagram
        autonumber
        actor User as Client User
        participant API as API Gateway
        participant Auth as Auth Service
        participant DB as User Database

        User->>API: POST /login (credentials)
        activate API
        API->>Auth: Validate Token()
        activate Auth
        Auth->>DB: QueryUser(email)
        DB-->>Auth: User Record & Hash
        Auth-->>API: 200 OK + JWT
        deactivate Auth
        API-->>User: 200 OK (JWT Session)
        deactivate API
    "#;
    let lines = DiagramRenderer::render_to_lines("mermaid", body, 120);
    assert!(!lines.is_empty());
    let text = lines_to_text(&lines);
    assert!(text.contains("Client User") || text.contains("User"));
    assert!(text.contains("API Gateway") || text.contains("API"));
    assert!(text.contains("Auth Service") || text.contains("Auth"));
}

#[test]
fn test_intermediate_02_class_diagram() {
    let body = r#"
    classDiagram
        class Animal {
            +String name
            +int age
            +makeSound() void
        }
        class Dog {
            +String breed
            +bark() void
        }
        class Cat {
            +bool indoor
            +meow() void
        }
        Animal <|-- Dog : Inherits
        Animal <|-- Cat : Inherits
    "#;
    let lines = DiagramRenderer::render_to_lines("mermaid", body, 120);
    assert!(!lines.is_empty());
    let text = lines_to_text(&lines);
    assert!(text.contains("Animal"));
    assert!(text.contains("Dog"));
    assert!(text.contains("Cat"));
}

#[test]
fn test_intermediate_03_state_diagram() {
    let body = r#"
    stateDiagram-v2
        [*] --> Idle
        Idle --> Processing : EvSubmit
        Processing --> Verifying : EvProcessDone
        Verifying --> Approved : EvPass
        Verifying --> Rejected : EvFail
        Approved --> [*]
        Rejected --> [*]
    "#;
    let lines = DiagramRenderer::render_to_lines("mermaid", body, 120);
    assert!(!lines.is_empty());
    let text = lines_to_text(&lines);
    assert!(text.contains("Idle"));
    assert!(text.contains("Processing"));
    assert!(text.contains("Approved") || text.contains("Done"));
}

#[test]
fn test_intermediate_04_er_diagram() {
    let body = r#"
    erDiagram
        CUSTOMER ||--o{ ORDER : places
        ORDER ||--|{ LINE-ITEM : contains
        CUSTOMER {
            string id PK
            string name
            string email
        }
        ORDER {
            int order_id PK
            string customer_id FK
            float total_amount
        }
    "#;
    let lines = DiagramRenderer::render_to_lines("mermaid", body, 120);
    assert!(!lines.is_empty());
    let text = lines_to_text(&lines);
    assert!(text.contains("CUSTOMER") || text.contains("Customer"));
    assert!(text.contains("ORDER") || text.contains("Order"));
}

#[test]
fn test_intermediate_05_gantt_chart() {
    let body = r#"
    gantt
        title Release 2.0 Roadmap
        dateFormat YYYY-MM-DD
        section Core Development
            Engine Optimization :done, t1, 2026-01-01, 2026-01-15
            SIMD Vector Kernels :active, t2, 2026-01-16, 2026-01-30
        section UI & Polish
            TUI Layout Redesign :crit, t3, 2026-02-01, 10d
    "#;
    let lines = DiagramRenderer::render_to_lines("mermaid", body, 120);
    assert!(!lines.is_empty());
    let text = lines_to_text(&lines);
    assert!(text.contains("Roadmap") || text.contains("Release") || text.contains("Optimization") || text.contains("Development"));
}

#[test]
fn test_intermediate_06_pie_chart() {
    let body = r#"
    pie title Compute Backend Distribution
        "AVX-512 Kernels" : 45
        "AVX2 Fused GEMV" : 35
        "Scalar Safe Core" : 20
    "#;
    let lines = DiagramRenderer::render_to_lines("mermaid", body, 100);
    assert!(!lines.is_empty());
    let text = lines_to_text(&lines);
    assert!(text.contains("AVX-512") || text.contains("Compute Backend Distribution"));
}

#[test]
fn test_intermediate_07_git_graph() {
    let body = r#"
    gitGraph
        commit id: "Initial commit"
        commit id: "Add engine core"
        branch feature/simd
        checkout feature/simd
        commit id: "SIMD Gemv"
        checkout main
        merge feature/simd
        commit id: "Release v1.0"
    "#;
    let lines = DiagramRenderer::render_to_lines("mermaid", body, 120);
    assert!(!lines.is_empty());
    let text = lines_to_text(&lines);
    assert!(text.contains("commit") || text.contains("Initial") || text.contains("simd") || text.contains("Release"));
}

// =========================================================================
// LEVEL 3: PROFESSIONAL PRODUCTION ARCHITECTURES & SCHEMATICS
// =========================================================================

#[test]
fn test_professional_01_cloud_native_microservices_architecture() {
    let body = r#"
    graph TD
        Client[Public Web & Mobile App] --> CDN[Cloudflare Edge CDN]
        CDN --> Ingress[NGINX Ingress Controller]
        
        subgraph Gateway [API Gateway Layer]
            Ingress --> APIGW[Kong Gateway & Rate Limiter]
        end

        subgraph CoreServices [Microservices Mesh]
            APIGW --> AuthSvc[Authentication Service]
            APIGW --> OrderSvc[Order Processing Service]
            APIGW --> UserSvc[User Profile Service]
            APIGW --> PaySvc[Payment Gateway Service]
        end

        subgraph EventBus [Event Streaming]
            OrderSvc --> Kafka[Apache Kafka Cluster]
            PaySvc --> Kafka
            Kafka --> WorkerSvc[Async Fulfillment Workers]
        end

        subgraph StorageTier [High Availability Databases]
            AuthSvc --> RedisAuth[(Redis Session Cache)]
            UserSvc --> PostgresUser[(PostgreSQL Primary)]
            OrderSvc --> PostgresOrder[(PostgreSQL Sharded)]
            WorkerSvc --> S3[(AWS S3 Document Lake)]
        end
    "#;
    let lines = DiagramRenderer::render_to_lines("mermaid", body, 140);
    assert!(!lines.is_empty());
    let text = lines_to_text(&lines);
    assert!(text.contains("Client") || text.contains("Mobile"));
    assert!(text.contains("Kong Gateway") || text.contains("APIGW"));
    assert!(text.contains("Kafka") || text.contains("Event"));
    assert!(text.contains("PostgreSQL") || text.contains("Postgres"));
}

#[test]
fn test_professional_02_kubernetes_deployment_topology() {
    let body = r#"
    graph TD
        Helm[Helm Chart Release] --> K8sAPI[Kubernetes Control Plane API]
        K8sAPI --> Deploy[Hercules Agent Deployment]
        Deploy --> RS[ReplicaSet - 3 Replicas]
        RS --> Pod1[Pod: hercules-01]
        RS --> Pod2[Pod: hercules-02]
        RS --> Pod3[Pod: hercules-03]
        
        Pod1 --> PVC1[(PersistentVolumeClaim 50GB)]
        Pod2 --> PVC2[(PersistentVolumeClaim 50GB)]
        Pod3 --> PVC3[(PersistentVolumeClaim 50GB)]
        
        Svc[ClusterIP Service: hercules-svc:8080] --> Pod1
        Svc --> Pod2
        Svc --> Pod3
    "#;
    let lines = DiagramRenderer::render_to_lines("mermaid", body, 140);
    assert!(!lines.is_empty());
    let text = lines_to_text(&lines);
    assert!(text.contains("Helm Chart") || text.contains("Helm"));
    assert!(text.contains("Kubernetes") || text.contains("K8sAPI"));
    assert!(text.contains("hercules-01") || text.contains("Pod"));
}

#[test]
fn test_professional_03_circuit_schematic_pin_preservation() {
    let body = r#"
    graph TD
        A[NE555 Timer Chip] -->|Output| B[N-Channel MOSFET Gate]
        B -->|Drain| C["12V Motor (775)"]
        B -->|Source| D[Ground Plane]
        E[Resistor R1] -->|To Pin 1| A
        F[Resistor R2] -->|To Pin 2| A
        G[Capacitor C1] -->|To Pin 6| A
        H[Capacitor C2] -->|To Pin 7| A
        I[Power Supply 12V] -->|To Pin 8| A
        J[Master Switch] -->|Control Input| A
        B -->|MOSFET Drain| C
        style A fill:#f9d966,stroke:#333
        style B fill:#33ccff,stroke:#333
        style C fill:#ff9933,stroke:#333
        style D fill:#666666,stroke:#333
    "#;
    let lines = DiagramRenderer::render_to_lines("mermaid", body, 160);
    assert!(!lines.is_empty());
    let text = lines_to_text(&lines);
    println!("RENDERED CIRCUIT TEXT:\n{}", text);
    assert!(text.contains("NE555 Timer"));
    assert!(text.contains("MOSFET"));
    assert!(text.contains("12V Motor") || text.contains("775"));
    assert!(!text.contains("CoTorPinI1put"));
}

#[test]
fn test_professional_04_ci_cd_pipeline_workflow() {
    let body = r#"
    graph LR
        Push[Git Push Event] --> Lint[Static Analysis & Clippy]
        Lint --> Test[Unit & Integration Tests]
        Test --> Bench[Benchmark Suite]
        Bench --> Build[Docker Multi-Stage Build]
        Build --> SecScan[Trivy Vulnerability Scan]
        SecScan --> PushReg[Push Image to Registry]
        PushReg --> HelmDeploy[ArgoCD GitOps Sync]
        HelmDeploy --> Canary[Canary Deployment (10%)]
        Canary --> FullProd[Full Production Rollout]
    "#;
    let lines = DiagramRenderer::render_to_lines("mermaid", body, 280);
    assert!(!lines.is_empty());
    let text = lines_to_text(&lines);
    assert!(text.contains("Push") || text.contains("Git"));
    assert!(text.contains("Static Analysis") || text.contains("Lint") || text.contains("Clippy"));
    assert!(text.contains("Docker") || text.contains("Build") || text.contains("Canary") || text.contains("Test"));
}

#[test]
fn test_professional_05_malformed_llm_syntax_sanitization() {
    // Tests messy unquoted parens, stray classDef/click directives, and unclosed tags
    let messy_body = r#"
        %% Comments that should be stripped
        graph TD
        NodeA[Main Controller (v2.4.1)] -->|Signal (Active-High)| NodeB[Sensor Array (I2C)]
        NodeB --> NodeC[EEPROM Memory (24LC256)]
        click NodeA "https://example.com" "Tooltip"
        style NodeA fill:#f9f,stroke:#333,stroke-width:4px
        classDef default fill:#111,stroke:#333
        class NodeB default
    "#;
    let lines = DiagramRenderer::render_to_lines("mermaid", messy_body, 140);
    assert!(!lines.is_empty());
    let text = lines_to_text(&lines);
    assert!(text.contains("Main Controller") || text.contains("Controller"));
    assert!(text.contains("Sensor Array") || text.contains("Sensor"));
    assert!(text.contains("EEPROM Memory") || text.contains("EEPROM"));
}

// =========================================================================
// SMALL GRAPH UI TESTS
// =========================================================================

#[test]
fn test_small_graph_ui_two_nodes() {
    let body = r#"
    graph TD
        A[Input] --> B[Output]
    "#;
    let lines = DiagramRenderer::render_mermaid(body, 80);
    assert!(!lines.is_empty());
    let text = lines_to_text(&lines);
    assert!(text.contains("Input"));
    assert!(text.contains("Output"));
    // Verify block corners ▛, ▜, ▌, ▐, ▙, ▟
    assert!(text.contains('▛') && text.contains('▜'));
    assert!(text.contains('▌') && text.contains('▐'));
    assert!(text.contains('▙') && text.contains('▟'));
    assert!(text.contains("──► [Output]"));
}

#[test]
fn test_small_graph_ui_with_edge_label() {
    let body = r#"
    graph TD
        A[Start] -->|Click| B[Process]
        B -->|Finish| C[Done]
    "#;
    let lines = DiagramRenderer::render_mermaid(body, 80);
    let text = lines_to_text(&lines);
    assert!(text.contains("Start"));
    assert!(text.contains("Process"));
    assert!(text.contains("Done"));
    assert!(text.contains("|Click|"));
    assert!(text.contains("|Finish|"));
    assert!(text.contains("──► [Process]"));
    assert!(text.contains("──► [Done]"));
}

#[test]
fn test_small_graph_ui_styling_and_alignment() {
    let body = r#"
    graph TD
        A[Sensor] -->|Data| B[Actuator]
        style A fill:#f9d966,stroke:#333
        style B fill:#33ccff,stroke:#333
    "#;
    let lines = DiagramRenderer::render_mermaid(body, 80);
    assert!(!lines.is_empty());

    // Verify fill and stroke colors are applied to spans
    let has_yellow = lines.iter().any(|l| {
        l.spans.iter().any(|s| {
            s.style.bg == Some(ratatui::style::Color::Rgb(249, 217, 102))
                || s.style.fg == Some(ratatui::style::Color::Rgb(249, 217, 102))
        })
    });
    let has_cyan = lines.iter().any(|l| {
        l.spans.iter().any(|s| {
            s.style.bg == Some(ratatui::style::Color::Rgb(51, 204, 255))
                || s.style.fg == Some(ratatui::style::Color::Rgb(51, 204, 255))
        })
    });
    assert!(has_yellow, "Sensor node yellow fill/stroke styling should be applied");
    assert!(has_cyan, "Actuator node cyan fill/stroke styling should be applied");

    // Verify box width alignment: top, middle, and bottom rows of node A have identical character counts
    let top_row = lines.iter().find(|l| lines_to_text(&[(*l).clone()]).contains('▛')).unwrap();
    let mid_row = lines.iter().find(|l| lines_to_text(&[(*l).clone()]).contains("Sensor")).unwrap();
    let bot_row = lines.iter().find(|l| lines_to_text(&[(*l).clone()]).contains('▙')).unwrap();

    let top_len = top_row.spans.iter().map(|s| s.content.chars().count()).sum::<usize>();
    let mid_len = mid_row.spans.iter().map(|s| s.content.chars().count()).sum::<usize>();
    let bot_len = bot_row.spans.iter().map(|s| s.content.chars().count()).sum::<usize>();

    assert_eq!(top_len, mid_len, "Top border and middle content row must have matching character width");
    assert_eq!(mid_len, bot_len, "Middle content row and bottom border must have matching character width");
}
