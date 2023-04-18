import curses

from sklearn.metrics import roc_curve, auc, f1_score, matthews_corrcoef

import matplotlib.pyplot as plt

import numpy as np


class WordlistEditor:
    """
    An editor to edit wordlists and their predicted mappings on the terminal.
    """

    def __init__(self, mapper, wordlist, solutions=None, validations=None, mappings=None):
        """
        Initializes a wordlist editor

        :param mapper: Mapper used to generate mapping predictions
        :type mapper: mapping.Mapper
        :param wordlist: A list of words to generate mappings for
        :type wordlist: [str]
        :param solutions: A list of correct mappings for each word in the word list
        :type solutions: [str]|None
        :param validations: A list of boolean values describing whether or not each mapping has been validated
        :type validations: [bool|None]|None
        :param mappings: A list of pre-predicted mappings (useful when importing previous workstates)
        :type mappings: [str]|None
        """
        self._mapper = mapper
        self._wordlist = wordlist
        self._embeddings = []
        self._mappings = mappings or [""] * len(wordlist)
        self._validations = validations or [None] * len(wordlist)
        """:type: [bool|None]"""
        self._solutions = solutions or ([None] * len(wordlist))

        self._current_index = 0
        self._running = False

    def start(self):
        """Launches the editor and its main loop."""

        print("Generating mappings...")
        self.generate_mappings()

        self._running = True
        curses.wrapper(self.run)

    def generate_mappings(self):
        """
        Generates the embedding vectors for all words in the wordlist and mappings if they have not been specified yet.
        """

        has_mappings = self._mappings[0] != ''

        for index, word in enumerate(self._wordlist):
            self._embeddings.append(self._mapper.model.embed_term(word))
            if not has_mappings:
                mappings, _ = self._mapper.predict_embedded(self._embeddings[index], 1)
                self._mappings[index] = mappings[0]
                if mappings[0] == self._solutions[index]:
                    self._validations[index] = True

    def run(self, stdscr):
        """
        Runs the editor's main loop by showing the current entry and waiting for the next command.

        :param stdscr: The curses window object to be used to display the editor in
        :type stdscr: curses.Window
        """
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_RED, -1)

        while self._running:
            self.clear_screen(stdscr)
            self.render_current_entry(stdscr)

            # Wait for command
            ch = stdscr.getch()
            self.handle_input(stdscr, ch)

    def handle_input(self, stdscr, ch):
        """
        Handles input in the editor's main loop, i.e. processes and forwards any entered commands.

        :param stdscr: The curses window object to be used to display the editor in
        :type stdscr: curses.Window
        :param ch: The key which was pressed
        :type ch: int
        """
        if ch == curses.ERR:
            return

        if ch == curses.KEY_LEFT:
            if self._current_index > 0:
                self._current_index -= 1
        elif ch == curses.KEY_RIGHT:
            if self._current_index < len(self._wordlist) - 1:
                self._current_index += 1
        elif ch == curses.KEY_DOWN:
            target = self._current_index
            while target < len(self._wordlist) - 1:
                target += 1
                if not self._validations[target]:
                    break
            self._current_index = target
        elif ch == curses.KEY_UP:
            target = self._current_index
            while target > 0:
                target -= 1
                if not self._validations[target]:
                    break
            self._current_index = target
        elif ch == ord('j') or ch == ord('J'):
            self.jump_to_entry(stdscr)
        elif ch == ord('a') or ch == ord('A'):
            self.preview_alternatives(stdscr)
        elif ch == ord('s') or ch == ord('S'):
            self.search_embedding(stdscr)
        elif ch == ord('w') or ch == ord('W'):
            self.write_to_file(stdscr)
        elif ch == ord('v') or ch == ord('V'):
            self.validate_entry(True)
        elif ch == ord('f') or ch == ord('F'):
            self.validate_entry(False)
        elif ch == ord('q') or ch == ord('Q'):
            self._running = False
        elif ch == ord('e') or ch == ord('E'):
            self.visualize_results(stdscr)
        elif ch == ord('h') or ch == ord('H'):
            self.render_help(stdscr)
            stdscr.getch()

    def clear_screen(self, stdscr):
        """
        Clears the terminal screen.

        :param stdscr: The curses window object to be used to display the editor in
        :type stdscr: curses.Window
        """
        stdscr.clear()
        stdscr.move(0, 0)

    def render_help(self, stdscr):
        """
        Renders a help text of available commands.

        :param stdscr: The curses window object to be used to display the editor in
        :type stdscr: curses.Window
        """
        self.clear_screen(stdscr)
        stdscr.addstr("Edit mapping using the following commands:\n")
        stdscr.addstr("Left  - Previous entry\n")
        stdscr.addstr("Right - Next entry\n")
        stdscr.addstr("Up    - Previous unvalidated entry\n")
        stdscr.addstr("Down  - Next unvalidated entry\n")
        stdscr.addstr("J     - Jump to entry\n")
        stdscr.addstr("A     - Display alternative mappings\n")
        stdscr.addstr("S     - Search embedding\n")
        stdscr.addstr("W     - Write to CSV file\n")
        stdscr.addstr("V/F   - Validate/Invalidate current entry (F)\n")
        stdscr.addstr("E     - Export visualizations / statistics\n")
        stdscr.addstr("Q     - Quit\n")
        stdscr.addstr("\n")
        stdscr.addstr("Press any key to return to editor\n")
        stdscr.refresh()

    def render_current_entry(self, stdscr):
        """
        Renders the currently displayed entry from the wordlist.

        :param stdscr: The curses window object to be used to display the editor in
        :type stdscr: curses.Window
        """
        color_pair = curses.color_pair(0)
        if self._validations[self._current_index] is not None:
            if self._validations[self._current_index]:
                color_pair = curses.color_pair(1)
            else:
                color_pair = curses.color_pair(2)

        word = self._wordlist[self._current_index]
        mapping = self._mappings[self._current_index]

        stdscr.addstr("Press H for a list of available commands.")
        stdscr.addstr("\n")
        stdscr.addstr("Entry:    {} of {}\n".format(self._current_index + 1, len(self._wordlist)))
        stdscr.addstr("Term:     {}\n".format(word))
        stdscr.addstr("Mapping:  ")
        stdscr.addstr("{}\n".format(mapping), color_pair)

        solution = self._solutions[self._current_index]
        if solution is not None:
            stdscr.addstr("Solution: {}\n".format(self._solutions[self._current_index]))
            if solution != mapping:
                stdscr.addstr("\n")
                stdscr.addstr("Note: when visualizing, validated mappings which differ from the specified solutions"
                              "will still be counted as true positives.\n")

    def render_options(self, stdscr, options):
        """
        Renders a list of up to 10 options the user can choose from and returns the index of the option the user chose.

        :param stdscr: The curses window object to be used to display the editor in
        :type stdscr: curses.Window
        :param options: A list of options to choose from (maximum of ten!)
        :type options: [str]
        :return: The index of the option the user chose
        :rtype: int
        """
        for i in range(min(len(options), 10)):
            stdscr.addstr("{} - {}\n".format(i, options[i]))
        stdscr.addstr("\n")
        stdscr.addstr("Choose alternative using 0 - {}\n".format(min(len(options), 10) - 1))

        index = None
        while index is None:
            ch = stdscr.getch()
            if ord('0') <= ch <= ord('9'):
                digit = ch - ord('0')
                if digit < len(options):
                    index = digit

        return index

    def get_input(self, stdscr, prompt):
        """
        Asks the user for a line of input, then returns that input.

        :param stdscr: The curses window object to be used to display the editor in
        :type stdscr: curses.Window
        :param prompt: The prompt to display to the user when asking for input
        :type prompt: str
        :return: The user's input
        :rtype: str
        """
        self.clear_screen(stdscr)
        stdscr.addstr(prompt + "\n")
        curses.echo()
        (cx, cy) = stdscr.getyx()
        input = stdscr.getstr(cy + 1, 0)
        curses.noecho()
        return input.decode()

    def jump_to_entry(self, stdscr):
        """
        Asks the user for an entry number to jump to, then changes the current entry accordingly.

        :param stdscr: The curses window object to be used to display the editor in
        :type stdscr: curses.Window
        """
        input = self.get_input(stdscr, "Please enter the index of the entry you wish to jump to or 0 to abort")
        next_index = None
        try:
            next_index = int(input) - 1
        except ValueError as verr:
            return
        except Exception as ex:
            return

        if next_index < 0 or next_index >= len(self._wordlist):
            return

        self._current_index = next_index

    def preview_alternatives(self, stdscr):
        """
        Displays a list of alternative mapping predictions for a specific entry and allows the user to
        change the mapping to one of them.

        :param stdscr: The curses window object to be used to display the editor in
        :type stdscr: curses.Window
        """
        ALTERNATIVES = 5

        word = self._wordlist[self._current_index]
        mappings, _ = self._mapper.predict(word, n_guesses=ALTERNATIVES)
        self.clear_screen(stdscr)
        stdscr.addstr("Term: {}\n".format(word))
        stdscr.addstr("\n")
        index = self.render_options(stdscr, mappings)
        self._mappings[self._current_index] = mappings[index]

    def search_embedding(self, stdscr):
        """
        Allows the user to search the embedding for a more appropriate mapping than the ones found in
        an entry's alternatives.

        :param stdscr: The curses window object to be used to display the editor in
        :type stdscr: curses.Window
        """
        mappings, _ = self._mapper.predict(self.get_input(stdscr, "Enter a query string:"), n_guesses=5)

        self.clear_screen(stdscr)
        stdscr.addstr("Term: {}\n".format(self._wordlist[self._current_index]))
        stdscr.addstr("\n")
        index = self.render_options(stdscr, mappings)
        self._mappings[self._current_index] = mappings[index]

    def validate_entry(self, default):
        """
        Toggles the validation status of the current entry.

        :param default: The validation status to assign an entry which has not been (in-)validated before.
        :type default: bool
        """
        if self._validations[self._current_index] is None:
            self._validations[self._current_index] = default
        else:
            self._validations[self._current_index] = not self._validations[self._current_index]

    def write_to_file(self, stdscr):
        """
        Exports the current wordlist, mappings, solutions and validations to a CSV file.

        :param stdscr: The curses window object to be used to display the editor in
        :type stdscr: curses.Window
        """
        path = self.get_input(stdscr, "Enter file path to write to:")

        try:
            with open(path, 'w') as f:
                f.write("Term#Mapping#Solution#Valid\n")
                for i in range(len(self._wordlist)):
                    f.write("{}#{}".format(self._wordlist[i], self._mappings[i]))
                    if self._solutions[i] is None:
                        f.write("#")
                    else:
                        f.write("#{}".format(self._solutions[i]))

                    if self._validations[i] is None:
                        f.write("#")
                    else:
                        f.write("#{}".format(self._validations[i]))

                    f.write("\n")
                f.flush()
        except Exception as e:
            pass

    def visualize_results(self, stdscr):
        """
        Calculates several metrics and a ROC curve for the current wordlist and validations.

        :param stdscr: The curses window object to be used to display the editor in
        :type stdscr: curses.Window
        """
        self.clear_screen(stdscr)

        # check if solutions or validations have been provided for all terms:
        num_words = len(self._wordlist)
        final_solutions = [''] * num_words
        for i in range(num_words):
            if self._validations[i]:
                final_solutions[i] = self._mappings[i]
            else:
                final_solutions[i] = self._solutions[i]

            if not final_solutions[i]:
                stdscr.addstr("No solution available for term {}, please validate manually.\n".format(i+1))
                stdscr.getch()
                return

        self.calculate_f1_scores(stdscr, final_solutions)
        stdscr.addstr("\n")
        self.calculate_mcc(stdscr, final_solutions)
        self.generate_roc_curve(stdscr, final_solutions)
        stdscr.addstr("\n")
        stdscr.addstr("Press any key to continue...\n")
        stdscr.getch()

    def generate_roc_curve(self, stdscr, final_solutions):
        """
        Generates a ROC curve and displays it.

        :param stdscr: The curses window object to be used to display the editor in
        :type stdscr: curses.Window
        :param final_solutions: The list of finalized solutions, i.e. 'correct mappings' for the wordlist
        """
        estimator = self._mapper.get_estimator()

        n_classes = len(estimator.classes_)
        n_samples = len(final_solutions)

        # calculate probabilities of positive class for all terms in wordlist:
        y_score = estimator.predict_proba(self._embeddings)

        counts = np.zeros(n_classes)
        for solution in final_solutions:
            search = np.where(estimator.classes_ == solution)
            if len(search[0]) == 0:
                continue
            index = search[0][0]
            counts[index] += 1

        label = dict()
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        weight = dict()

        n_classes_in_samples = 0

        for i in range(n_classes):
            if counts[i] == 0:
                continue

            positive_class = estimator.classes_[i]
            label[n_classes_in_samples] = positive_class
            fpr[n_classes_in_samples], tpr[n_classes_in_samples], _ = roc_curve(final_solutions, y_score[:, i], pos_label=positive_class)
            roc_auc[n_classes_in_samples] = auc(fpr[n_classes_in_samples], tpr[n_classes_in_samples])
            weight[n_classes_in_samples] = counts[i] / n_samples

            n_classes_in_samples += 1

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes_in_samples)]))

        summed_tpr = np.zeros_like(all_fpr)
        weighted_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes_in_samples):
            # For each class there could be a different number of ROC data points
            # to average we therefore need to interpolate the missing points
            interpolated = np.interp(all_fpr, fpr[i], tpr[i])
            weighted_tpr += interpolated * weight[i]
            summed_tpr += interpolated

        fpr["macro"] = all_fpr
        tpr["macro"] = summed_tpr / n_classes_in_samples
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        fpr["weighted"] = all_fpr
        tpr["weighted"] = weighted_tpr
        roc_auc["weighted"] = auc(fpr["weighted"], tpr["weighted"])

        plt.figure()
        plt.plot(fpr["macro"], tpr["macro"], color="navy", lw=2, label="macro ROC curve (area = %0.2f)" % roc_auc["macro"])
        plt.plot(fpr["weighted"], tpr["weighted"], color="darkorange", lw=2, label="weighted ROC curve (area = %0.2f)" % roc_auc["weighted"])
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC statistics")
        plt.legend(loc="lower right")
        plt.show()

    def calculate_f1_scores(self, stdscr, final_solutions):
        """
        Calculates the F1 scores using macro, micro and weighted averaging and displays them.

        :param stdscr: The curses window object to be used to display the editor in
        :type stdscr: curses.Window
        :param final_solutions: The list of finalized solutions, i.e. 'correct mappings' for the wordlist
        """
        estimator = self._mapper.get_estimator()
        f1_micro = f1_score(final_solutions, self._mappings, labels=estimator.classes_, average='micro', zero_division=0)
        f1_macro = f1_score(final_solutions, self._mappings, labels=estimator.classes_, average='macro', zero_division=0)
        f1_weighted = f1_score(final_solutions, self._mappings, labels=estimator.classes_, average='weighted', zero_division=0)

        stdscr.addstr("F1-Scores:\n")
        stdscr.addstr("\tmicro:    {}\n".format(f1_micro))
        stdscr.addstr("\tmacro:    {}\n".format(f1_macro))
        stdscr.addstr("\tweighted: {}\n".format(f1_weighted))

    def calculate_mcc(self, stdscr, final_solutions):
        """
        Calculates the Matthews Correlation Coefficient and displays it.

        :param stdscr: The curses window object to be used to display the editor in
        :type stdscr: curses.Window
        :param final_solutions: The list of finalized solutions, i.e. 'correct mappings' for the wordlist
        """
        mcc = matthews_corrcoef(final_solutions, self._mappings)
        stdscr.addstr("Matthews Correlation Coefficient: {}\n".format(mcc))
