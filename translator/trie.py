class TrieNode():
    def __init__(self):
        # Initialising one node for trie
        self.children = {}
        self.last = False
        self.loc = -1
 
 
class Trie():
    def __init__(self):
 
        # Initialising the trie structure.
        self.root = TrieNode()
        self.templist = []
 
    def formTrie(self, keys):
 
        # Forms a trie structure with the given set of strings
        # if it does not exists already else it merges the key
        # into it by extending the structure as required
        for key in keys:
            self.insert(key.lower())  # inserting one key to the trie.
 
    def insert(self, key, x=-1):
 
        # Inserts a key into trie if it does not exist already.
        # And if the key is a prefix of the trie node, just
        # marks it as leaf node.
        node = self.root
 
        for a in key:
            if not node.children.get(a):
                node.children[a] = TrieNode()
 
            node = node.children[a]
 
        node.last = True
        node.loc = x
 
    def find(self, word):

        # search fro word in trie, returns loc if found 

        node = self.root

        for a in word:
            if not node.children[a]:
                return False
            node = node.children[a]
        
        return node.loc

    # def suggestionsRec(self, node, word):
 
    #     # Method to recursively traverse the trie
    #     # and return a whole word.

    #     self.templist = [] 

    #     if node.last:
    #         print(word)
    #         return word
 
    #     for a, n in node.children.items():
    #         self.suggestionsRec(n, word + a)


    def suggestionsRec(self, node, word):
 
        # Method to recursively traverse the trie
        # and return a whole word.

        self.templist = [] 

        if node.last:
            # print(word)
            self.templist.append(word)
 
        for a, n in node.children.items():
            self.templist += self.suggestionsRec(n, word + a)
        
        return self.templist

 
    def getAutoSuggestions(self, key):
 
        # Returns all the words in the trie whose common
        # prefix is the given key thus listing out all
        # the suggestions for autocomplete.
        node = self.root
 
        for a in key:
            # no string in the Trie has this prefix
            if not node.children.get(a):
                return 0
            node = node.children[a]
 
        # If prefix is present as a word, but
        # there is no subtree below the last
        # matching node.
        if not node.children:
            return -1
 
        return self.suggestionsRec(node, key)