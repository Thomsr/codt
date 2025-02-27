pub trait Instance {
    fn id(&self) -> i32;

    fn read(id: i32, line: String) -> Self;
}

pub struct ClassificationInstance {
    id: i32,
    #[allow(dead_code)]
    label: i32,
}

impl ClassificationInstance {
    pub fn new(label: i32) -> Self {
        Self { id: 0, label }
    }
}

impl Instance for ClassificationInstance {
    fn id(&self) -> i32 {
        self.id
    }

    fn read(id: i32, line: String) -> Self {
        let parts: Vec<&str> = line.split(' ').collect();
        let label = parts
            .first()
            .expect("Expected at least a label at this line");
        let label: i32 = label
            .parse()
            .expect("Expected an integer label at the start of the line");

        Self { id, label }
    }
}
