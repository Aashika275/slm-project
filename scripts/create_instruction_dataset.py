import os
import json
import random


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "instruction_tuning"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "codex_instruction_train.jsonl"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 100 UNIQUE INSTRUCTION EXAMPLES
# ============================================================

examples = [

# ============================================================
# PYTHON - 20
# ============================================================

{
"instruction": "Write a Python function to calculate the factorial of a number.",
"response": """def factorial(n):
    if n < 0:
        raise ValueError("n must be non-negative")

    result = 1

    for i in range(2, n + 1):
        result *= i

    return result"""
},

{
"instruction": "Write a Python function that checks whether a number is prime.",
"response": """def is_prime(n):
    if n < 2:
        return False

    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False

    return True"""
},

{
"instruction": "Write Python code to find the second largest number in a list.",
"response": """def second_largest(numbers):
    unique_numbers = list(set(numbers))

    if len(unique_numbers) < 2:
        raise ValueError("At least two unique numbers are required")

    unique_numbers.sort(reverse=True)

    return unique_numbers[1]"""
},

{
"instruction": "Write a Python function to check whether two strings are anagrams.",
"response": """def are_anagrams(a, b):
    return sorted(a.replace(" ", "").lower()) == \
           sorted(b.replace(" ", "").lower())"""
},

{
"instruction": "Write Python code to calculate the Fibonacci sequence.",
"response": """def fibonacci(n):
    sequence = []

    a, b = 0, 1

    for _ in range(n):
        sequence.append(a)
        a, b = b, a + b

    return sequence"""
},

{
"instruction": "Write a Python function to find duplicate values in a list.",
"response": """def find_duplicates(items):
    seen = set()
    duplicates = set()

    for item in items:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)

    return list(duplicates)"""
},

{
"instruction": "Write Python code to sort a list of dictionaries by age.",
"response": """people = [
    {"name": "Arun", "age": 25},
    {"name": "Priya", "age": 21},
    {"name": "Kumar", "age": 30}
]

people.sort(key=lambda person: person["age"])

print(people)"""
},

{
"instruction": "Explain Python list comprehension with an example.",
"response": """List comprehension provides a compact way to create lists.

Example:

numbers = [1, 2, 3, 4, 5]

squares = [n * n for n in numbers]

print(squares)

The expression loops through numbers and creates a new list containing each squared value."""
},

{
"instruction": "Write Python code to read a JSON file.",
"response": """import json

with open("data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

print(data)"""
},

{
"instruction": "Write Python code to write a dictionary to a JSON file.",
"response": """import json

data = {
    "name": "CODEX",
    "version": 1
}

with open("data.json", "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4)"""
},

{
"instruction": "Explain Python generators.",
"response": """A generator produces values one at a time instead of storing the complete result in memory.

The yield keyword is used to produce values.

Example:

def numbers():
    for i in range(3):
        yield i

for number in numbers():
    print(number)

Generators are useful when processing large amounts of data."""
},

{
"instruction": "Write a Python function that safely converts a string to an integer.",
"response": """def to_integer(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None"""
},

{
"instruction": "Write Python code to count vowels in a string.",
"response": """def count_vowels(text):
    vowels = set("aeiou")
    return sum(
        1 for char in text.lower()
        if char in vowels
    )"""
},

{
"instruction": "Explain the difference between shallow copy and deep copy in Python.",
"response": """A shallow copy creates a new outer object but keeps references to nested objects.

A deep copy recursively copies nested objects as well.

Example:

import copy

shallow = copy.copy(original)
deep = copy.deepcopy(original)

Deep copies are useful when nested mutable objects must be completely independent."""
},

{
"instruction": "Write Python code to find the most common element in a list.",
"response": """from collections import Counter

def most_common(items):
    if not items:
        return None

    return Counter(items).most_common(1)[0][0]"""
},

{
"instruction": "Write a Python function to flatten a list of lists.",
"response": """def flatten(items):
    result = []

    for group in items:
        for item in group:
            result.append(item)

    return result"""
},

{
"instruction": "Explain Python decorators with a simple example.",
"response": """A decorator is a function that modifies or extends another function without changing its source code.

Example:

def log_call(func):
    def wrapper():
        print("Function called")
        return func()
    return wrapper

@log_call
def hello():
    print("Hello")

hello()"""
},

{
"instruction": "Write Python code to merge two dictionaries.",
"response": """first = {"a": 1, "b": 2}
second = {"c": 3, "d": 4}

merged = {**first, **second}

print(merged)"""
},

{
"instruction": "Write Python code to handle a file-not-found exception.",
"response": """try:
    with open("config.json", "r") as file:
        content = file.read()
except FileNotFoundError:
    print("Configuration file was not found.")"""
},

{
"instruction": "Explain Python virtual environments.",
"response": """A virtual environment creates an isolated Python environment for a project.

It allows a project to install its own package versions without affecting other projects.

Example:

python -m venv .venv

Activate it on Windows PowerShell:

.venv\\Scripts\\Activate.ps1"""
},


# ============================================================
# JAVASCRIPT / TYPESCRIPT - 15
# ============================================================

{
"instruction": "Write JavaScript code to find the largest number in an array.",
"response": """function largest(numbers) {
    return Math.max(...numbers);
}

console.log(largest([4, 8, 2, 10]));"""
},

{
"instruction": "Write JavaScript code to remove duplicate values from an array.",
"response": """function removeDuplicates(values) {
    return [...new Set(values)];
}

console.log(removeDuplicates([1, 2, 2, 3, 3]));"""
},

{
"instruction": "Explain JavaScript promises.",
"response": """A Promise represents the eventual result of an asynchronous operation.

A promise can be pending, fulfilled, or rejected.

Example:

fetch("/api/users")
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error(error));"""
},

{
"instruction": "Write JavaScript code to check whether an array contains a value.",
"response": """const numbers = [10, 20, 30];

const exists = numbers.includes(20);

console.log(exists);"""
},

{
"instruction": "Explain async and await in JavaScript.",
"response": """async and await provide a readable way to work with Promises.

An async function returns a Promise.

await pauses execution inside the async function until the Promise settles.

Example:

async function loadData() {
    const response = await fetch("/api/data");
    const data = await response.json();
    return data;
}"""
},

{
"instruction": "Write JavaScript code to calculate the sum of an array.",
"response": """function sum(numbers) {
    return numbers.reduce(
        (total, number) => total + number,
        0
    );
}"""
},

{
"instruction": "Explain JavaScript event bubbling.",
"response": """Event bubbling means an event triggered on a child element can propagate upward through its parent elements.

For example, clicking a button inside a div can trigger the button handler and then the div handler.

event.stopPropagation() can be used when propagation should be stopped."""
},

{
"instruction": "Write JavaScript code to validate that an email field is not empty.",
"response": """function validateEmail(email) {
    if (!email || email.trim() === "") {
        return false;
    }

    return true;
}"""
},

{
"instruction": "Explain the difference between == and === in JavaScript.",
"response": """== performs equality comparison with type coercion.

=== performs strict equality comparison without converting types.

For example:

5 == "5"   // true
5 === "5"  // false

Using === is generally preferred when strict type comparison is intended."""
},

{
"instruction": "Write JavaScript code to filter numbers greater than 10.",
"response": """const numbers = [5, 12, 8, 20, 3];

const result = numbers.filter(
    number => number > 10
);

console.log(result);"""
},

{
"instruction": "Create a TypeScript type for a product.",
"response": """type Product = {
    id: number;
    name: string;
    price: number;
    inStock: boolean;
};"""
},

{
"instruction": "Write a TypeScript function that adds two numbers.",
"response": """function add(a: number, b: number): number {
    return a + b;
}"""
},

{
"instruction": "Explain TypeScript generics.",
"response": """Generics allow TypeScript code to work with different types while preserving type information.

Example:

function identity<T>(value: T): T {
    return value;
}

const numberValue = identity<number>(10);
const textValue = identity<string>("hello");"""
},

{
"instruction": "Create a TypeScript interface for an API response.",
"response": """interface ApiResponse<T> {
    success: boolean;
    data: T;
    message?: string;
}"""
},

{
"instruction": "Write JavaScript code to convert an object into JSON.",
"response": """const user = {
    name: "Arun",
    age: 25
};

const json = JSON.stringify(user);

console.log(json);"""
},


# ============================================================
# SQL / DATABASE - 15
# ============================================================

{
"instruction": "Write SQL to retrieve all products with price greater than 1000.",
"response": """SELECT *
FROM products
WHERE price > 1000;"""
},

{
"instruction": "Write SQL to update the price of a product.",
"response": """UPDATE products
SET price = 1500
WHERE id = 10;"""
},

{
"instruction": "Write SQL to delete a user by ID.",
"response": """DELETE FROM users
WHERE id = 10;"""
},

{
"instruction": "Explain the purpose of a SQL foreign key.",
"response": """A foreign key creates a relationship between tables.

It references a key in another table and helps maintain referential integrity.

Example:

FOREIGN KEY (user_id)
REFERENCES users(id)"""
},

{
"instruction": "Write SQL to retrieve the five most expensive products.",
"response": """SELECT *
FROM products
ORDER BY price DESC
LIMIT 5;"""
},

{
"instruction": "Write SQL to calculate the average product price.",
"response": """SELECT AVG(price) AS average_price
FROM products;"""
},

{
"instruction": "Explain the SQL GROUP BY clause.",
"response": """GROUP BY groups rows that have the same values in specified columns.

It is commonly used with aggregate functions such as COUNT, SUM, AVG, MIN, and MAX.

Example:

SELECT category, COUNT(*)
FROM products
GROUP BY category;"""
},

{
"instruction": "Write SQL to find users who have placed at least one order.",
"response": """SELECT DISTINCT u.id, u.name
FROM users u
JOIN orders o
    ON u.id = o.user_id;"""
},

{
"instruction": "Explain database indexing.",
"response": """A database index is a data structure that helps the database locate rows more efficiently.

Indexes can significantly improve read performance for frequently searched columns.

However, indexes require storage and can increase the cost of INSERT, UPDATE, and DELETE operations."""
},

{
"instruction": "Write SQL to create an index on an email column.",
"response": """CREATE INDEX idx_users_email
ON users(email);"""
},

{
"instruction": "Explain database normalization.",
"response": """Normalization organizes data to reduce unnecessary duplication and improve data consistency.

Common normal forms include first normal form, second normal form, and third normal form.

The appropriate level depends on the application's requirements and query patterns."""
},

{
"instruction": "Write SQL to find duplicate email addresses.",
"response": """SELECT email, COUNT(*) AS occurrences
FROM users
GROUP BY email
HAVING COUNT(*) > 1;"""
},

{
"instruction": "Explain the difference between DELETE and TRUNCATE.",
"response": """DELETE removes rows from a table and can include a WHERE condition.

TRUNCATE removes all rows from a table and is generally intended for quickly clearing the table.

Exact transactional and identity-reset behavior depends on the database system."""
},

{
"instruction": "Write SQL to join customers and orders using customer_id.",
"response": """SELECT
    customers.name,
    orders.id,
    orders.total
FROM customers
JOIN orders
    ON customers.id = orders.customer_id;"""
},

{
"instruction": "Explain a database transaction.",
"response": """A transaction groups multiple database operations into one logical unit.

A transaction should preserve consistency according to the database's transaction guarantees.

Typical operations are:

BEGIN;
perform operations;
COMMIT;

If a failure occurs, ROLLBACK can undo the transaction's changes."""
},


# ============================================================
# BACKEND / API - 15
# ============================================================

{
"instruction": "Create an Express.js POST endpoint for creating a product.",
"response": """const express = require("express");

const app = express();

app.use(express.json());

app.post("/api/products", (req, res) => {
    const { name, price } = req.body;

    if (!name || price === undefined) {
        return res.status(400).json({
            message: "Name and price are required"
        });
    }

    res.status(201).json({
        name,
        price
    });
});"""
},

{
"instruction": "Explain REST API resource naming conventions.",
"response": """REST APIs commonly use nouns to represent resources.

Examples:

GET /users
GET /users/10
POST /users
PUT /users/10
DELETE /users/10

Using resource-oriented URLs keeps APIs consistent and easier to understand."""
},

{
"instruction": "Explain HTTP status code 400.",
"response": """HTTP 400 Bad Request indicates that the server could not process the request because the request was invalid.

It can be used for malformed input, invalid parameters, or failed request validation."""
},

{
"instruction": "Explain HTTP status code 401 versus 403.",
"response": """401 indicates that authentication is required or the supplied authentication credentials are not acceptable.

403 indicates that the server understood the request but the authenticated client is not permitted to perform the requested operation."""
},

{
"instruction": "Write Express.js middleware that logs request methods.",
"response": """function requestLogger(req, res, next) {
    console.log(
        `${req.method} ${req.originalUrl}`
    );

    next();
}

app.use(requestLogger);"""
},

{
"instruction": "Explain JWT authentication at a high level.",
"response": """JWT authentication commonly works by issuing a signed token after successful authentication.

The client sends the token with later requests.

The server verifies the token's signature and claims before allowing protected operations.

JWTs should be handled carefully, including secure storage, expiration, and appropriate authorization checks."""
},

{
"instruction": "Write an Express.js error-handling middleware.",
"response": """function errorHandler(err, req, res, next) {
    console.error(err);

    res.status(500).json({
        message: "Internal server error"
    });
}

app.use(errorHandler);"""
},

{
"instruction": "Explain request validation in backend APIs.",
"response": """Request validation checks incoming data before business logic or database operations are performed.

It should verify required fields, data types, formats, ranges, and application-specific constraints.

Validation helps prevent invalid data and reduces unexpected application behavior."""
},

{
"instruction": "Explain the purpose of environment variables in backend applications.",
"response": """Environment variables allow configuration values to be supplied outside application source code.

Common examples include:

DATABASE_URL
PORT
API_KEY

Sensitive values should not be hard-coded into source files or committed to public repositories."""
},

{
"instruction": "Explain pagination in a REST API.",
"response": """Pagination divides a large result set into smaller pages.

A request might use:

GET /products?page=2&limit=20

The server returns a limited number of records and can provide metadata such as total records or available pages."""
},

{
"instruction": "Explain CORS in web applications.",
"response": """CORS stands for Cross-Origin Resource Sharing.

It controls whether a browser allows web pages from one origin to make requests to another origin.

Servers configure CORS response headers to specify which origins, methods, and headers are allowed."""
},

{
"instruction": "Write a simple Express.js health-check endpoint.",
"response": """app.get("/health", (req, res) => {
    res.status(200).json({
        status: "ok"
    });
});"""
},

{
"instruction": "Explain API rate limiting.",
"response": """Rate limiting restricts how many requests a client can make within a given period.

It helps protect APIs from excessive traffic, accidental request loops, and certain forms of abuse.

Common implementations use counters or token-bucket-style algorithms."""
},

{
"instruction": "Explain the purpose of HTTP PUT and PATCH.",
"response": """PUT is commonly used to replace or fully update a resource.

PATCH is commonly used to apply a partial update.

For example:

PUT /users/10

can represent a complete replacement, while:

PATCH /users/10

can change only selected fields."""
},

{
"instruction": "Describe a basic backend request flow.",
"response": """A typical backend request flow is:

1. Client sends HTTP request.
2. Router identifies the endpoint.
3. Middleware performs cross-cutting tasks.
4. Controller validates input.
5. Service layer executes business logic.
6. Repository or database layer accesses data.
7. Server returns an HTTP response.

The exact structure depends on the application's architecture."""
},


# ============================================================
# GIT / GITHUB - 10
# ============================================================

{
"instruction": "Explain git clone.",
"response": """git clone downloads a remote Git repository to the local machine.

Example:

git clone https://github.com/example/project.git

It creates a local working copy containing the repository history."""
},

{
"instruction": "Explain git status.",
"response": """git status shows the current state of the working tree.

It can show modified files, staged files, untracked files, and the current branch."""
},

{
"instruction": "Explain git add.",
"response": """git add moves changes into the staging area.

Example:

git add app.py

To stage all changed files:

git add ."""
},

{
"instruction": "Explain git commit.",
"response": """git commit records staged changes in the Git repository history.

Example:

git commit -m "Add user authentication"

A useful commit message briefly describes the change."""
},

{
"instruction": "Explain git merge.",
"response": """git merge integrates the history of one branch into another.

For example, while on main:

git merge feature-login

Git may perform a fast-forward merge or create a merge commit depending on the branch history."""
},

{
"instruction": "Explain a Git merge conflict.",
"response": """A merge conflict occurs when Git cannot automatically reconcile changes from different branches.

The developer must inspect the conflicting files, decide which content should remain, remove conflict markers, and then stage and commit the resolved files."""
},

{
"instruction": "Give commands to rename the current Git branch to main.",
"response": """git branch -M main
git push -u origin main

The first command renames the current local branch. The second pushes main and establishes the upstream relationship."""
},

{
"instruction": "Explain git log.",
"response": """git log displays commit history.

Useful examples:

git log
git log --oneline
git log --oneline --graph

The compact form is useful for quickly inspecting repository history."""
},

{
"instruction": "Explain .gitignore.",
"response": """.gitignore tells Git which files or directories should normally not be tracked.

Common examples include:

node_modules/
.venv/
.env
__pycache__/
*.log

It helps keep generated files and sensitive configuration out of the repository."""
},

{
"instruction": "Explain why .env files should usually not be committed.",
"response": """.env files often contain secrets such as database passwords, API keys, or access tokens.

Committing them can expose credentials to other users or attackers.

Sensitive configuration should instead be managed through environment variables or a secure secret-management system."""
},


# ============================================================
# DOCKER / DEPLOYMENT - 10
# ============================================================

{
"instruction": "Explain docker build.",
"response": """docker build creates a Docker image from a Dockerfile and its build context.

Example:

docker build -t myapp .

The -t option gives the image a name and optional tag."""
},

{
"instruction": "Explain docker run.",
"response": """docker run creates and starts a container from a Docker image.

Example:

docker run -p 5000:5000 myapp

The -p option maps a host port to a container port."""
},

{
"instruction": "Explain Docker port mapping.",
"response": """Docker port mapping connects a host port to a container port.

Example:

docker run -p 8080:5000 myapp

This means requests sent to port 8080 on the host are forwarded to port 5000 inside the container."""
},

{
"instruction": "Explain Docker volumes.",
"response": """Docker volumes provide persistent storage for containers.

Container filesystems can disappear when containers are removed, while a volume can preserve important data.

Volumes are commonly used for databases and other persistent application data."""
},

{
"instruction": "Explain Docker Compose.",
"response": """Docker Compose allows multiple related containers and their configuration to be defined together.

A Compose file can describe services, networks, volumes, ports, and environment variables.

It is useful for local multi-service development."""
},

{
"instruction": "Write a basic docker-compose service for PostgreSQL.",
"response": """services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: appuser
      POSTGRES_PASSWORD: apppassword
      POSTGRES_DB: appdb
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:"""
},

{
"instruction": "Explain why Docker containers should not normally run unnecessary services.",
"response": """Keeping containers focused on a single primary responsibility simplifies deployment, monitoring, debugging, and scaling.

A multi-service container can make lifecycle management and failure diagnosis more complicated."""
},

{
"instruction": "Explain the purpose of a Docker health check.",
"response": """A Docker health check allows a container's application health to be tested using a configured command.

It helps determine whether the application inside the container is actually functioning rather than merely having a running container process."""
},

{
"instruction": "Explain Docker image tags.",
"response": """An image tag identifies a particular version or variant of a Docker image.

Examples:

myapp:1.0
myapp:2.0
myapp:latest

Explicit version tags are useful for reproducible deployments."""
},

{
"instruction": "List basic steps for deploying a backend application.",
"response": """A basic deployment process is:

1. Build the application.
2. Configure environment variables.
3. Build the deployment artifact or Docker image.
4. Provision the target server or platform.
5. Configure networking and security.
6. Start the application.
7. Run health checks.
8. Monitor logs and application metrics."""
},


# ============================================================
# TESTING / QA - 5
# ============================================================

{
"instruction": "Explain regression testing.",
"response": """Regression testing verifies that existing functionality still works after code changes.

It helps detect unintended side effects introduced by new features, bug fixes, or refactoring."""
},

{
"instruction": "Write a simple pytest test for a multiply function.",
"response": """def multiply(a, b):
    return a * b


def test_multiply():
    assert multiply(3, 4) == 12"""
},

{
"instruction": "Explain test-driven development.",
"response": """Test-driven development, or TDD, commonly follows:

1. Write a failing test.
2. Implement the minimum code needed to pass it.
3. Refactor the implementation.
4. Repeat.

The cycle is often called Red, Green, Refactor."""
},

{
"instruction": "Explain mocking in software testing.",
"response": """Mocking replaces a real dependency with a controlled test object.

For example, an API test may mock a database service so the test focuses on business logic without requiring a real database.

Mocks can control return values and verify interactions."""
},

{
"instruction": "List useful test cases for a user registration API.",
"response": """Useful test cases include:

1. Valid registration.
2. Missing required fields.
3. Invalid email.
4. Weak password.
5. Duplicate email.
6. Very long input.
7. Invalid data types.
8. Database failure.
9. Unauthorized access where applicable.
10. Successful response format and status code."""
},


# ============================================================
# SECURITY - 5
# ============================================================

{
"instruction": "Explain the principle of least privilege.",
"response": """The principle of least privilege means giving users, applications, and services only the permissions they need to perform their required tasks.

Reducing unnecessary permissions limits the potential impact of mistakes or compromised accounts."""
},

{
"instruction": "Explain why HTTPS is important.",
"response": """HTTPS encrypts HTTP communication using TLS.

It helps protect data from interception and provides authentication of the server when certificates are correctly configured.

It is important for protecting credentials, session data, and other sensitive information in transit."""
},

{
"instruction": "Explain secure password hashing.",
"response": """Password hashing should use a password-specific hashing algorithm such as Argon2 or bcrypt.

Passwords should not be encrypted and stored in a reversible form.

A unique salt and appropriate work factor help make offline password guessing more difficult."""
},

{
"instruction": "List common web application security risks.",
"response": """Common risks include:

1. Injection vulnerabilities.
2. Broken authentication.
3. Broken authorization.
4. Cross-site scripting.
5. Cross-site request forgery.
6. Security misconfiguration.
7. Sensitive data exposure.
8. Insecure dependency usage."""
},

{
"instruction": "Explain why API keys should not be hard-coded in source code.",
"response": """Hard-coded API keys can accidentally be exposed through source repositories, logs, or distributed application packages.

Configuration should be supplied through environment variables, deployment secrets, or an appropriate secret-management system."""
},


# ============================================================
# ARCHITECTURE / DEBUGGING - 5
# ============================================================

{
"instruction": "Explain separation of concerns in software architecture.",
"response": """Separation of concerns means dividing a system so different responsibilities are handled by appropriate components.

For example, user-interface code, business logic, database access, and authentication can be separated.

This can improve maintainability, testing, and code organization."""
},

{
"instruction": "Explain the repository pattern.",
"response": """The repository pattern separates data-access operations from business logic.

A service can request data through a repository instead of directly containing database queries.

This can make data access easier to test and replace."""
},

{
"instruction": "A backend server starts but the frontend cannot connect. List debugging steps.",
"response": """Check:

1. The backend port.
2. The frontend API URL.
3. Whether the backend is listening on the expected host.
4. Browser network errors.
5. CORS configuration.
6. Firewall rules.
7. Proxy configuration.
8. Backend logs.
9. Whether the requested endpoint actually exists."""
},

{
"instruction": "Explain logging best practices for backend applications.",
"response": """Backend logs should provide useful information about application behavior without exposing secrets.

Useful practices include structured logging, appropriate log levels, timestamps, request identifiers, error context, and avoiding passwords, tokens, and other sensitive values in logs."""
},

{
"instruction": "Describe a basic layered backend architecture.",
"response": """A layered backend can contain:

1. Routes — map HTTP requests to handlers.
2. Controllers — handle request and response logic.
3. Services — contain business logic.
4. Repositories — handle data access.
5. Database — persist application data.

The exact number of layers can vary depending on project complexity."""
}

]


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 60)
print("VALIDATING DATASET")
print("=" * 60)

instructions = [
    item["instruction"]
    for item in examples
]

responses = [
    item["response"]
    for item in examples
]

print()
print("Total examples:", len(examples))
print("Unique instructions:", len(set(instructions)))
print("Unique responses:", len(set(responses)))

if len(examples) != 100:
    raise RuntimeError(
        f"Expected exactly 100 examples, found {len(examples)}"
    )

if len(set(instructions)) != 100:
    raise RuntimeError(
        "Duplicate instructions detected."
    )

if len(set(responses)) != 100:
    raise RuntimeError(
        "Duplicate responses detected."
    )


# ============================================================
# SHUFFLE
# ============================================================

random.seed(42)

random.shuffle(examples)


# ============================================================
# WRITE JSONL
# ============================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    for item in examples:
        f.write(
            json.dumps(
                item,
                ensure_ascii=False
            )
            + "\n"
        )


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 60)
print("CODEX-SLM INSTRUCTION DATASET CREATED")
print("=" * 60)

print()
print("Output:")
print(OUTPUT_PATH)

print()
print("Total examples:")
print(len(examples))

print()
print("Unique instructions:")
print(len(set(instructions)))

print()
print("Unique responses:")
print(len(set(responses)))

print()
print("Dataset creation completed successfully.")
