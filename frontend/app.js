let books = [];

function addBook(){

    const input = document.getElementById("book");

    if(input.value.trim() !== ""){
        books.push(input.value);

        let li = document.createElement("li");
        li.textContent = input.value;

        document.getElementById("bookList").appendChild(li);

        input.value = "";
    }
}

function generateRecommendations(){

    document.getElementById("results").style.display="block";

    // const recommendations = [

    //     {
    //         title:"The Hobbit",
    //         author:"J.R.R. Tolkien",
    //         affinity:"96%"
    //     },

    //     {
    //         title:"Dune",
    //         author:"Frank Herbert",
    //         affinity:"94%"
    //     },

    //     {
    //         title:"Foundation",
    //         author:"Isaac Asimov",
    //         affinity:"92%"
    //     },

    //     {
    //         title:"The Da Vinci Code",
    //         author:"Dan Brown",
    //         affinity:"89%"
    //     }

    // ];
    fetch(
        "http://localhost:5000/recommend",
        {
        method:"POST",
        headers:{
            "Content-Type":"application/json"
        },
        body: JSON.stringify({
            books: books
        })
        }
    )
    .then(res=>res.json())
    .then(data=>{

        let html="";

        data.forEach(book=>{

        html += `
        <div class="book-card">
            <h3>${book.title}</h3>
            <p>Afinidad: ${(book.score*100).toFixed(1)}%</p>
        </div>
        `;
        });

        document
        .getElementById("recommendations")
        .innerHTML = html;
    });
    let html = "";

    recommendations.forEach(book => {

        html += `
        <div class="book-card">

            <h3>${book.title}</h3>

            <p>Autor: ${book.author}</p>

            <p>Afinidad estimada: ${book.affinity}</p>

        </div>`;
    });

    document.getElementById("recommendations").innerHTML = html;
}