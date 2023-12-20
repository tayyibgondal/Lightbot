/**
 * Class representing a simple chatbox interface.
 */
class Chatbox {
  /**
   * Create a Chatbox instance.
   */
  constructor() {
    /**
     * @property {Object} args - The arguments containing DOM elements.
     * @property {HTMLElement} args.openButton - The button to open/close the chatbox.
     * @property {HTMLElement} args.chatBox - The container for the chatbox.
     * @property {HTMLElement} args.sendButton - The button to send a message.
     */
    this.args = {
      openButton: document.querySelector(".chatbox__button"),
      chatBox: document.querySelector(".chatbox__support"),
      sendButton: document.querySelector(".send__button"),
    };

    /**
     * @property {boolean} state - The state of the chatbox (open or closed).
     */
    this.state = false;
  
    /**
     * @property {Array} messages - An array to store chat messages.
     * Each message is an object with properties 'name' and 'message'.
     */
    this.messages = [];
  }

  /**
   * Display the chatbox and set up event listeners.
   */
  display() {
    const { openButton, chatBox, sendButton } = this.args;

    openButton.addEventListener("click", () => this.toggleState(chatBox));

    sendButton.addEventListener("click", () => this.onSendButton(chatBox));

    const node = chatBox.querySelector("input");
    node.addEventListener("keyup", ({ key }) => {
      if (key == "Enter") {
        this.onSendButton(chatBox);
      }
    });
  }

  /**
   * Toggle the state of the chatbox (open or closed).
   * @param {HTMLElement} chatbox - The chatbox container element.
   */
  toggleState(chatbox) {
    this.state = !this.state;

    if (this.state) {
      chatbox.classList.add("chatbox--active");
    } else {
      chatbox.classList.remove("chatbox--active");
    }
  }

  /**
   * Handle sending a message from the user.
   * @param {HTMLElement} chatbox - The chatbox container element.
   */
  onSendButton(chatbox) {
    var textField = chatbox.querySelector("input");
    let text1 = textField.value;
    if (text1 == "") {
      return;
    }

    // Create a message object for the user's input.
    let msg1 = { name: "User", message: text1 };
    this.messages.push(msg1);

    // Send the user's message to the server for a response.
    fetch($SCRIPT_ROOT + "/predict", {
      method: "POST",
      body: JSON.stringify({ message: text1 }),
      mode: "cors",
      headers: {
        "Content-Type": "application/json",
      },
    })
      .then((r) => r.json())
      .then((r) => {
        // Display message back to the user
        let msg2 = { name: "Sam", message: r.answer };
        this.messages.push(msg2);
        this.updateChatText(chatbox);
        textField.value = "";
      }).catch((error) => {
        console.error('Error:', error);
        this.updateChatText(chatbox);
        textField.value = ''
      });
  }

  /**
   * Update the chatbox's message display.
   * @param {HTMLElement} chatbox - The chatbox container element.
   */
  updateChatText(chatbox) {
    var html = '';
    this.messages.slice().reverse().forEach(function(item,index) {
        if(item.name == "Sam") {
            html += '<div class="messages__item messages__item--visitor">' + item.message + "</div>";
        }
        else {
            html += '<div class="messages__item messages__item--operator">' + item.message + "</div>";
        }
    });

    const chatmessage = chatbox.querySelector('.chatbox__messages');
    chatmessage.innerHTML = html;
  }
}

// Create an instance of the Chatbox class and display the chatbox.
const chatbox = new Chatbox();
chatbox.display();
